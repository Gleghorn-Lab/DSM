import importlib.util
import torch


def is_compiled_module(module: torch.nn.Module) -> bool:
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def extract_model_from_parallel(model: torch.nn.Module, keep_torch_compile: bool = True) -> torch.nn.Module:
    parallel_wrappers = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    while isinstance(model, parallel_wrappers):
        model = model.module

    is_compiled = is_compiled_module(model)
    if is_compiled:
        compiled_model = model
        assert "_orig_mod" in dir(compiled_model), (
            "Expected compiled model to contain _orig_mod for unwrapping."
        )
        model = compiled_model._orig_mod

    if keep_torch_compile and is_compiled:
        compiled_model._orig_mod = model
        model = compiled_model

    return model


def patch_accelerate_extract_model_from_parallel() -> None:
    if importlib.util.find_spec("accelerate") is None:
        return

    import accelerate.utils
    import accelerate.utils.other

    accelerate.utils.extract_model_from_parallel = extract_model_from_parallel
    accelerate.utils.other.extract_model_from_parallel = extract_model_from_parallel

    if importlib.util.find_spec("transformers.modeling_utils") is None:
        return

    import transformers.modeling_utils as modeling_utils

    modeling_utils.extract_model_from_parallel = extract_model_from_parallel
