#! /usr/bin/env python3

# fmt: off
import torch
import argparse
import sys
import numpy as np

# import lemurian
# lemurian.lemurian_context().__enter__()

# Supported three-operand operations
OPS = {
    'clip': lambda a, u, d: torch.maximum(torch.minimum(a, u), d),
    'mac': lambda a, b, c: torch.fma(a, b, c) if hasattr(torch, 'fma') else a * b + c,
    'sel': lambda s, a, b: torch.where(s != 0.0, a, b),
}

parser = argparse.ArgumentParser(description='Run a ternary torch operation and save results.')
parser.add_argument('--op', choices=OPS.keys(), required=True,
                    help='Ternary operation to compute')
parser.add_argument('--file', type=str, default=None,
                    help='Input binary float file containing interleaved (a,b,c) triplets')
parser.add_argument('--size', type=int, default=201,
                    help='Number of random input triplets when --file is not given (default: 201)')
args = parser.parse_args()

op_fn = OPS[args.op]
op_name = args.op


def MODEL(a, b, c):
    return op_fn(a, b, c)


# Prepare inputs
if args.file:
    arr = np.fromfile(args.file, dtype=np.float32)
    if arr.size % 3 != 0:
        print('Input file must contain triplets of floats (a, b, c)')
        sys.exit(1)
    arr = arr.reshape(-1, 3)
    input_a = torch.from_numpy(arr[:, 0]).reshape(1, -1).to('cuda:0')
    input_b = torch.from_numpy(arr[:, 1]).reshape(1, -1).to('cuda:0')
    input_c = torch.from_numpy(arr[:, 2]).reshape(1, -1).to('cuda:0')
else:
    input_a = torch.randn([1, args.size], dtype=torch.float32, device='cuda:0')
    input_b = torch.randn([1, args.size], dtype=torch.float32, device='cuda:0')
    input_c = torch.randn([1, args.size], dtype=torch.float32, device='cuda:0')

# Get baseline result from torch
torcheager = MODEL(input_a, input_b, input_c)

# Run first compile and execution
compiled_module = torch.compile(MODEL)
torchinductor = compiled_module(input_a, input_b, input_c)

# Check that the results are the same
max_error = torch.max(torch.abs(torcheager - torchinductor)).item()
if torch.allclose(torcheager, torchinductor, atol=1e-06):
    print(f"\x1b[0;32;40m Torch {op_name} results and Diluvian results are the same: Max Error = {max_error:.8f} \033[0m")
else:
    print(f"Torch {op_name} results and Diluvian results differ: Max Error = {max_error:.8f}")

print(input_a)
print(input_b)
print(input_c)
print(torcheager)
print(torchinductor)

# Save as raw binary (include op name in filenames)
eager_file = f'torcheager{op_name}.bin'
inductor_file = f'torchinductor{op_name}.bin'
torcheager.to('cpu').numpy().tofile(eager_file)
torchinductor.to('cpu').numpy().tofile(inductor_file)
print(f"Wrote {eager_file} and {inductor_file}")

# vim: et ts=4 sw=4
