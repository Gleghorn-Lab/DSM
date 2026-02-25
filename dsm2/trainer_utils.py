import os
import torch
import torch.distributed as dist
from typing import Any, List, Tuple


def infer_rank_world_size_local_rank() -> Tuple[int, int, int]:
    rank = 0
    world_size = 1
    local_rank = 0

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

    return rank, world_size, local_rank


def gather_object_across_ranks(local_object: Any, is_distributed: bool, world_size: int) -> List[Any]:
    if not is_distributed:
        return [local_object]

    gathered_objects = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_objects, local_object)
    return gathered_objects


def reduce_loss_sum_and_count(
    local_loss_sum: float,
    local_loss_count: int,
    device: torch.device,
    is_distributed: bool,
) -> Tuple[float, int]:
    stats = torch.tensor([local_loss_sum, float(local_loss_count)], dtype=torch.float64, device=device)
    if is_distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item())


def reduce_mean_float(value: float, device: torch.device, is_distributed: bool, world_size: int) -> float:
    tensor_value = torch.tensor([value], dtype=torch.float64, device=device)
    if is_distributed:
        dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
        tensor_value = tensor_value / float(world_size)
    return float(tensor_value[0].item())
