import torch


def is_compiled_module(module: torch.nn.Module) -> bool:
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def extract_model_from_parallel(model: torch.nn.Module, keep_torch_compile: bool = True) -> torch.nn.Module:
    parallel_wrappers = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    is_compiled = is_compiled_module(model)
    if is_compiled:
        compiled_model = model
        model = model._orig_mod

    while isinstance(model, parallel_wrappers):
        model = model.module

    if keep_torch_compile and is_compiled:
        compiled_model._orig_mod = model
        model = compiled_model

    return model
