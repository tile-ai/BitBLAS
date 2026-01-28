# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
import os
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from tvm import IRModule
from tvm.runtime import Module
from tvm.tir import Schedule
from bitblas.tl.base_hint import BaseTLHint
from bitblas.base.arch import TileDevice
from bitblas.base.utils import get_dummy_input_arrays
from bitblas.utils import (
    tensor_replace_dp4a,
    tensor_remove_make_int4,
    tensor_remove_make_int2,
    retrieve_func_from_module,
)
from bitblas.common import MAX_ERROR_MESSAGE_LENGTH
from bitblas.base.base_scheduler import BaseScheduler
from bitblas.tl.lower import tl_lower

logger = logging.getLogger(__name__)


class CompileResult:
    """
    Class to store the result of compilation
    """

    def __init__(self, config, kernel):
        self.config = config
        self.kernel = kernel
        self.latency = 1e9

    def profile(self, data_distribution="uniform"):
        profiler = self.kernel.get_profiler()
        latency = profiler.do_bench()
        return latency


def _apply_config(
    scheduler: BaseScheduler,
    config: BaseTLHint = None,
    arch: TileDevice = None,
) -> Optional[IRModule]:
    """
    find rules:
    case 1. if the main block has no reduce op, then use the Elementwise rule.
    case 2. if the config enabled tensorcore, then use the TensorCore rule.
    case 3. if any([t > 1 for t in config.reduce_thread]), we should use the InnerThread Reduction Rule.
    case 4. else we should use general reduction rule.
    """
    logger.debug("Scheduler Apply config {}".format(config))
    scheduled_func = scheduler.apply_config(config, arch)
    if scheduled_func is None:
        return None
    else:
        return tvm.IRModule.from_expr(scheduled_func)


def apply_and_build_parallel(scheduler,
                             configs,
                             arch,
                             num_repeats=3,
                             max_workers=10,
                             timeout=60,
                             data_distribution="uniform") -> CompileResult:
    cpresults = []

    max_workers = min(len(configs), os.cpu_count(), max_workers)

    # apply config in thread parallel
    _scheduled_ir_modules: List[Schedule] = []

    def _submit_config(f, c, a):
        try:
            scheduled_ir_module = _apply_config(f, c, a)
        except Exception as apply_schedule_error:
            logger.debug("Apply schedule failed: {}".format(apply_schedule_error))
            scheduled_ir_module = None
        return scheduled_ir_module

    with ThreadPoolExecutor(max_workers=max_workers) as _scheduler:
        futures = {_scheduler.submit(_submit_config, scheduler, config, arch) for config in configs}
        for future in as_completed(futures, timeout=timeout):
            _scheduled_ir_modules.append(future.result())

    # build in process parallel
    def _build(context):
        idx, mod, arch = context
        if mod is None:
            return idx, None, None

        config = configs[idx]
        assert config is not None

        # Check only have one function in the module
        if len(mod.functions) > 1:
            raise ValueError("Only support one function in the module")

        tl_prim_func = list(mod.functions.values())[0]

        @tilelang.jit(
            target=arch.target,
            pass_configs={
                tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            },
        )
        def _jit_entry():
            return tl_prim_func

        kernel = _jit_entry()
        return idx, kernel

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_build, (i, mod, arch)): i
            for i, mod in enumerate(_scheduled_ir_modules)
        }

        for future in as_completed(future_to_idx, timeout=timeout):
            idx = future_to_idx[future]
            assert idx <= len(_scheduled_ir_modules), "Index out of range"
            assert idx <= len(configs), "Index out of range"

            ir_module = _scheduled_ir_modules[idx]
            config = configs[idx]
            try:
                idx, kernel = future.result()
            except Exception as e:
                logger.error(f"Build failed for idx {idx}: {e}")
                continue
            cpresult = CompileResult(config, kernel)
            cpresults.append(cpresult)

    best = None
    best_latency = 1e9
    for cpresult in cpresults:
        config = cpresult.config
        latency = cpresult.profile()
        logger.info("Evaluation with config {}".format(config))
        logger.info("Time cost of this config: {:.3f} ms".format(latency))

        cpresult.latency = latency
        if latency < best_latency:
            best_latency = latency
            best = cpresult
    return cpresults, best


def apply_and_build(
    scheduler,
    configs,
    arch,
    parallel_build=False,
    data_distribution="uniform",
) -> Tuple[List[CompileResult], CompileResult]:
    max_workers = 10 if parallel_build else 1
    return apply_and_build_parallel(
        scheduler, configs, arch, max_workers=max_workers, data_distribution=data_distribution)
