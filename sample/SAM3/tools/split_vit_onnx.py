#!/usr/bin/env python3
"""
Split existing SAM3 ViT ONNX parts into finer 2-block chunks.

The existing parts (1-4) each contain 8 blocks. This script splits each into
4 sub-parts of 2 blocks each, producing ONNX files suitable for SOC's 950MB TPU.

Usage (inside tpu_mlir Docker):
  python3 split_vit_onnx.py --input_dir ../models/onnx --output_dir ../models/onnx_soc

Author: liheng.fang
Date: 2025-06-23
"""

import os
import sys
import argparse
from collections import defaultdict

import onnx
from onnx import helper, numpy_helper
import numpy as np


def parse_block_index(node_name):
    """Extract block index from node name like '/blocks.3/norm1/ReduceMean'."""
    for part in node_name.split('/'):
        if part.startswith('blocks.') and part.split('.')[1].isdigit():
            return int(part.split('.')[1])
    return -1


def find_block_boundaries(graph):
    """
    Find the start and end nodes for each block in the graph.

    Returns:
        dict: block_idx -> {'start_nodes': [...], 'end_nodes': [...],
                             'all_nodes': [...], 'initializers': [...]}
    """
    blocks = defaultdict(lambda: {'nodes': [], 'init_names': set()})

    for node in graph.node:
        blk_idx = parse_block_index(node.name)
        if blk_idx >= 0:
            blocks[blk_idx]['nodes'].append(node)

    # Map initializers to blocks
    node_inputs = set()
    for blk_idx, data in blocks.items():
        for node in data['nodes']:
            for inp in node.input:
                node_inputs.add(inp)

    for init in graph.initializer:
        if init.name in node_inputs:
            # Find which block uses this initializer
            for blk_idx, data in blocks.items():
                for node in data['nodes']:
                    if init.name in node.input:
                        data['init_names'].add(init.name)
                        break

    return blocks


def find_block_input_output(graph, blocks, blk_idx):
    """
    Find the input and output tensor names for a specific block.

    Each block typically has:
    - Input: the output of the previous block (or the initial reshape)
    - Output: blocks.N/Add_1 (the final residual add)
    """
    blk_nodes = blocks[blk_idx]['nodes']

    # Find the block's final Add (residual connection)
    output_name = f'/blocks.{blk_idx}/Add_1'
    blk_output = None
    for node in blk_nodes:
        if node.name == output_name:
            blk_output = node.output[0]
            break

    # The block input is the first real compute node's input
    # that comes from outside the block
    all_blk_outputs = set()
    for node in blk_nodes:
        for out in node.output:
            all_blk_outputs.add(out)

    blk_input = None
    for node in blk_nodes:
        for inp in node.input:
            if inp not in all_blk_outputs and not inp.startswith(f'/blocks.{blk_idx}'):
                if inp not in graph.initializer:
                    blk_input = inp
                    break
        if blk_input:
            break

    return blk_input, blk_output


def extract_block_range(graph, blocks, start_blk, end_blk, part_idx,
                         output_dir, input_shape=(1, 5184, 1024)):
    """
    Extract a range of blocks [start_blk, end_blk] into a new ONNX model.
    """
    # Collect all nodes in range
    selected_nodes = []
    selected_inits = set()

    for blk_idx in range(start_blk, end_blk + 1):
        selected_nodes.extend(blocks[blk_idx]['nodes'])
        selected_inits.update(blocks[blk_idx]['init_names'])

    # Build value_info for intermediate tensors
    # Collect all inputs and outputs of selected nodes
    all_inputs = set()
    all_outputs = set()
    for node in selected_nodes:
        for inp in node.input:
            all_inputs.add(inp)
        for out in node.output:
            all_outputs.add(out)

    # External input = inputs not produced by any selected node
    external_inputs = all_inputs - all_outputs

    # Find the LAST block's final output (last Add node's output)
    last_blk = end_blk
    last_add_node = None
    for node in reversed(selected_nodes):
        name = node.name
        # Match nodes belonging to the last block that are Add ops
        if f'blocks.{last_blk}' in name and node.op_type == 'Add':
            last_add_node = node
            break

    if last_add_node is None:
        raise RuntimeError(f"Cannot find output node for block {last_blk}")

    last_output = last_add_node.output[0]

    # Build new graph
    # Input: only the data flow features tensor (not weight initializers)
    # The data input is typically a Reshape output or named clearly
    init_names_in_graph = {i.name for i in graph.initializer}
    data_inputs = []
    for ext_in in sorted(external_inputs):
        if ext_in in init_names_in_graph:
            continue  # skip known initializers
        # Also skip obvious constant patterns
        if ext_in.startswith('/Constant') or ext_in.startswith('onnx::'):
            continue
        data_inputs.append(ext_in)

    new_inputs = [helper.make_tensor_value_info(
        inp, onnx.TensorProto.FLOAT, input_shape) for inp in data_inputs]

    new_outputs = [helper.make_tensor_value_info(
        last_output, onnx.TensorProto.FLOAT, input_shape)]

    # Collect initializers — include all that are referenced
    init_map = {i.name: i for i in graph.initializer}
    # Also add any external input that looks like a constant (has no producer)
    new_inits = []
    for name in sorted(selected_inits):
        if name in init_map:
            new_inits.append(init_map[name])
    # Also add constants from the original graph
    for node in graph.node:
        if node.op_type == 'Constant':
            for name in node.output:
                if name in all_inputs:
                    # This constant is used by selected nodes
                    pass  # Constants are handled differently

    # Build graph
    new_graph = helper.make_graph(
        nodes=selected_nodes,
        name=f'vit_blocks_{start_blk}_{end_blk}',
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=new_inits,
    )

    # Build model
    new_model = helper.make_model(new_graph, opset_imports=[helper.make_opsetid('', 14)])

    # Validate
    try:
        onnx.checker.check_model(new_model)
    except Exception as e:
        print(f"  WARNING: ONNX check failed: {e}")
        print(f"  Continuing anyway...")

    # Save
    output_path = os.path.join(output_dir, f'sam3_vit_part{part_idx}.onnx')
    onnx.save(new_model, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Part {part_idx}: blocks {start_blk}-{end_blk} → {output_path} ({size_mb:.1f} MB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Split SAM3 ViT ONNX into finer parts')
    parser.add_argument('--input_dir', default='../models/onnx',
                        help='Directory with existing part1-4 ONNX files')
    parser.add_argument('--output_dir', default='../models/onnx_soc',
                        help='Output directory for split ONNX files')
    parser.add_argument('--blocks_per_part', type=int, default=2,
                        help='Target blocks per output part')
    parser.add_argument('--execute', action='store_true',
                        help='Actually run the split')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SAM3 ViT ONNX Split (Fine-Grained for SOC)")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Blocks/part: {args.blocks_per_part}")
    print("=" * 60)

    # Copy part 0 (patch embedding) unchanged
    part0_src = os.path.join(args.input_dir, 'sam3_vit_part0.onnx')
    part0_dst = os.path.join(args.output_dir, 'sam3_vit_part0.onnx')

    if not args.execute:
        print("\n[DRY RUN] Use --execute to actually run the split.")
        print(f"\n  Part 0: copy {part0_src} → {part0_dst}")

        for part_idx in range(1, 5):
            src = os.path.join(args.input_dir, f'sam3_vit_part{part_idx}.onnx')
            base_blk = (part_idx - 1) * 8
            num_sub = 8 // args.blocks_per_part
            for sub in range(num_sub):
                start = base_blk + sub * args.blocks_per_part
                end = start + args.blocks_per_part - 1
                out_part = (part_idx - 1) * num_sub + sub + 1
                dst = os.path.join(args.output_dir, f'sam3_vit_part{out_part}.onnx')
                print(f"  Part {out_part}: blocks {start}-{end} ← {src}")
        return

    # Copy part 0
    if os.path.exists(part0_src):
        import shutil
        shutil.copy2(part0_src, part0_dst)
        print(f"Part 0: copied {part0_src} → {part0_dst}")
    else:
        print(f"Part 0: NOT FOUND: {part0_src}")

    # Process parts 1-4
    global_part_idx = 1  # Starting part index for block-containing parts

    for src_part_idx in range(1, 5):
        src_path = os.path.join(args.input_dir, f'sam3_vit_part{src_part_idx}.onnx')
        if not os.path.exists(src_path):
            print(f"Part {src_part_idx}: SKIP (not found: {src_path})")
            continue

        print(f"\nLoading {os.path.basename(src_path)}...")
        model = onnx.load(src_path)
        graph = model.graph
        print(f"  {len(graph.node)} nodes, {len(graph.initializer)} initializers")

        blocks = find_block_boundaries(graph)
        blk_indices = sorted(blocks.keys())
        print(f"  Blocks found: {blk_indices}")

        num_sub = 8 // args.blocks_per_part

        for sub in range(num_sub):
            start = sub * args.blocks_per_part
            end = start + args.blocks_per_part - 1
            print(f"\n  Extracting blocks {start}-{end}...")
            extract_block_range(
                graph, blocks, start, end,
                global_part_idx, args.output_dir
            )
            global_part_idx += 1

    print("\n" + "=" * 60)
    print(f"Split complete! {global_part_idx} parts total (0 to {global_part_idx-1})")
    print(f"Output: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
