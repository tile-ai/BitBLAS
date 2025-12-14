# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Literal
from tvm import IRModule
from tvm.ir import GlobalVar
from tvm.script import tir as T


# fmt: off
# TIR interleave weight impl-> 2D implementation
def tir_quant_compress_weight(
    N: int = 2,
    K: int = 16,
    input_dtype: Literal["int8", "int32"] = "int8",
    storage_dtype: Literal["int8", "int32"] = "int8",
    bits: int = 4,
):
    elems_per_byte = 8 // bits
    QK = K // elems_per_byte
    assert K % elems_per_byte == 0, "K must be divisible by 8/bits"

    @T.prim_func
    def quant_compress_weight(W: T.Buffer((N, K), input_dtype), QW: T.Buffer((N, QK),
                                                                             storage_dtype)):
        for ax0, ax1, ax2 in T.grid(N, QK, elems_per_byte):
            QW[ax0, ax1] = (QW[ax0, ax1] | (W[ax0, ax1 * elems_per_byte + ax2] << (bits * ax2)))

    return quant_compress_weight


# fmt: on


def select_implementation(
    M: int,
    N: int,
    input_dtype: Literal["int8", "int32"] = "int8",
    storage_dtype: Literal["int8", "int32"] = "int8",
    dequantize_bits: int = 4,
):
    func = tir_quant_compress_weight(
        N=M,
        K=N,
        input_dtype=input_dtype,
        storage_dtype=storage_dtype,
        bits=dequantize_bits,
    )
    mod = IRModule()
    mod.update_func(GlobalVar("main"), func)
    return mod
