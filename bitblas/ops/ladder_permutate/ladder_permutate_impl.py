# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas.gpu.matmul_analysis import (
    get_propagate_map,
    get_ladder_stage3_map,
)
from typing import Literal
from tvm import te, IRModule, DataType
from tvm.tir import IndexMap
from tilelang import language as T

def select_implementation(
    M: int,
    N: int,
    datatype: Literal["float16", "bfloat16", "int8", "e4m3_float8", "e5m2_float8"] = "float16",
    dequantize_bits: int = -1,
    storage_dtype: Literal["float16", "int8", "uint8", "int32", "uint32"] = "float16",
    propagate_kind: Literal["A", "B"] = "B",
    transpose_matrix: bool = False,
    transform_kind: int = 0,
    target_instruction: Literal["nvidia-mma"] = "nvidia-mma",
):
    if target_instruction != "nvidia-mma":
        raise ValueError("Currently only support nvidia-mma instruction")

    # This is trick to get the basic tile size for the current datatype
    # as for nvidia tensorcore instruction, the basic tile size is 16x16/16x32 for float16/int8
    l = r = 16  # noqa: E741
    if datatype in ["int8", "e4m3_float8", "e5m2_float8"]:
        l, r = 16, 32  # noqa: E741
    intra_index_map, _ = get_propagate_map(
        transpose_matrix, dtype=datatype, matrix_name=propagate_kind)

    ladder_stage3_map, _ = get_ladder_stage3_map(dtype=datatype)

    target_dtype = DataType(datatype)
    scaling_factor = 1
    if dequantize_bits > 0 and dequantize_bits < target_dtype.bits:
        scaling_factor = ((target_dtype.bits // dequantize_bits) * DataType(storage_dtype).bits //
                          target_dtype.bits)
        r = r // scaling_factor
        initial_indices = intra_index_map.initial_indices
        scaling_final_indices = intra_index_map.map_indices(initial_indices[:-1] +
                                                            [initial_indices[-1] * scaling_factor])
        scaling_final_indices = scaling_final_indices[:-1] + [
            scaling_final_indices[-1] // scaling_factor
        ]
        intra_index_map = IndexMap(
            initial_indices,
            scaling_final_indices,
            None,
        )

        initial_indices = ladder_stage3_map.initial_indices
        scaling_final_indices = ladder_stage3_map.map_indices(
            initial_indices[:-1] + [initial_indices[-1] * scaling_factor])
        scaling_final_indices = scaling_final_indices[:-1] + [
            scaling_final_indices[-1] // scaling_factor
        ]
        ladder_stage3_map = IndexMap(
            initial_indices,
            scaling_final_indices,
            None,
        )

    inp = te.placeholder((M, N // scaling_factor), name="inp", dtype=storage_dtype)
    args = [inp]

    assert transform_kind != 0, "Permute only apply when transform_kind >= 1"
 
    if transform_kind == 2:
        @T.prim_func
        def intra_warp_permutate_func(inp: T.Buffer((M, (N // scaling_factor)), dtype=storage_dtype), out: T.Buffer((M // l, (N // scaling_factor) // r, l, r), dtype=storage_dtype)):
            # with T.Kernel(1, ) as _:
                for i, j, ii ,jj in T.grid(M // l, (N // scaling_factor) // r, l, r):
                    nii, njj = intra_index_map.map_indices([ii, jj])
                    out[i, j, ii, jj] = inp[i * l + nii, j * r + njj]
        return IRModule.from_expr(intra_warp_permutate_func)

    if transform_kind == 3:
        arg = args[-1]

        ladder_stage3_map_inverse = ladder_stage3_map.inverse([l, r])

        @T.prim_func
        def ladder_stage3_permutate_func(inp: T.Buffer((M, (N // scaling_factor)), dtype=storage_dtype), out: T.Buffer((M // l, (N // scaling_factor) // r, l, r), dtype=storage_dtype)):
            # with T.Kernel(1, ) as _:
                for i, j, ii, jj in T.grid(M // l, (N // scaling_factor) // r, l, r):
                    # Keep TE semantics: intra-warp permutation then stage3 inverse.
                    nii, njj = ladder_stage3_map_inverse.map_indices([ii, jj])
                    pii, pjj = intra_index_map.map_indices([nii, njj])
                    out[i, j, ii, jj] = inp[i * l + pii, j * r + pjj]

        return IRModule.from_expr(ladder_stage3_permutate_func)
    args = [args[0], args[-1]]

    func = te.create_prim_func(args)

    return IRModule.from_expr(func)
