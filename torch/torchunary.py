#! /usr/bin/env python3

# fmt: off
import torch
import argparse
import sys
import numpy as np

#import lemurian
#lemurian.lemurian_context().__enter__()

# Supported operations
OPS = {
    'exp': torch.exp,
    'exp2': torch.exp2,
    'reciprocal': torch.reciprocal,
    'cos': torch.cos,
    'sin': torch.sin,
    'sqrt': torch.sqrt,
    'rsqrt': torch.rsqrt,
}

parser = argparse.ArgumentParser(description='Run a unary torch operation and save results.')
parser.add_argument('--op', choices=OPS.keys(), required=True,
                    help='Unary operation to compute (exp, cos, sin, sqrt, rsqrt)')
parser.add_argument('--file', type=str, default=None,
                    help='Input binary float file (if omitted, uses random inputs)')
parser.add_argument('--size', type=int, default=65,
                    help='Number of random inputs when --file is not given (default: 65)')
args = parser.parse_args()

op_fn = OPS[args.op]
op_name = args.op

def MODEL(a):
    return op_fn(a)

# Prepare inputs
if args.file:
    arr = np.fromfile(args.file, dtype=np.float32)
    input_a = torch.from_numpy(arr).reshape(1, -1).to("cuda:0")
else:
    input_a = torch.randn([1, args.size], dtype=torch.float32, device="cuda:0")

# Get baseline result from torch
torcheager = MODEL(input_a)

# Run first compile and execution
compiled_module = torch.compile(MODEL)
torchinductor = compiled_module(input_a)

# check that the results are the same
max_error = torch.max(torch.abs(torcheager - torchinductor)).item()
if torch.allclose(torcheager, torchinductor, atol=1e-06):
    print(f"\x1b[0;32;40m Torch {op_name} results and Diluvian results are the same: Max Error = {max_error:.8f} \033[0m")
else:
    print(f"Torch {op_name} results and Diluvian results differ: Max Error = {max_error:.8f}")

print(input_a)
print(torcheager)
print(torchinductor)

# Save as raw binary (include op name in filenames)
eager_file = f'torcheager{op_name}.bin'
inductor_file = f'torchinductor{op_name}.bin'
torcheager.to("cpu").numpy().tofile(eager_file)
torchinductor.to("cpu").numpy().tofile(inductor_file)
print(f"Wrote {eager_file} and {inductor_file}")

# vim: et ts=4 sw=4
