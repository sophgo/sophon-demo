# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch
import torch_tpu
import torch.distributed as dist


def init_distributed_group():
    r"""initialize sequence parallel group.
    """
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))

        options = torch_tpu.ProcessGroupSCCLOptions()
        torch_tpu.tpu.set_chip_map(options, use_rank_table=False)
        torch_tpu.tpu.set_device(rank)

        dist.init_process_group(
            backend="sccl",
            rank=rank,
            world_size=world_size,
            pg_options=options,
        )


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = (
            x.view(1, *x.shape[:scatter_dim], world_size, -1, *x.shape[scatter_dim+1:])
            .transpose(0, scatter_dim+1)
            .squeeze(scatter_dim+1)
            .contiguous()
        )
        outputs = torch.empty_like(inputs)
        torch.tpu.synchronize()
        dist.all_to_all_single(outputs, inputs)
        x = (
            outputs.unsqueeze(gather_dim+1)
            .transpose(0, gather_dim+1)
            .reshape(*outputs.shape[1:gather_dim+1], -1, *outputs.shape[gather_dim+2:])
        )
    return x


def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()
