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
    'atan2': torch.atan2,
    'cldiv': lambda a, b: torch.ceil(a / b),
    'copysign': torch.copysign,
    'div': torch.div,
    'dim': lambda a, b: torch.clamp(a - b, min=0),
    'fldiv': lambda a, b: torch.floor(a / b),
    'fmax': torch.fmax,
    'fmin': torch.fmin,
    'fmod': torch.fmod,
    'hypot': torch.hypot,
    'nextafter': torch.nextafter,
    'pow': torch.pow,
    'remainder': torch.remainder,
    'root': lambda a, b: torch.pow(a, 1.0 / b),
}

parser = argparse.ArgumentParser(description='Run a binary torch operation and save results.')
parser.add_argument('--op', choices=OPS.keys(), required=True,
                    help='Binary operation to compute')
parser.add_argument('--file', type=str, default=None,
                    help='Input binary float file containing interleaved (a,b) pairs')
parser.add_argument('--size', type=int, default=201,
                    help='Number of random input pairs when --file is not given (default: 201)')
args = parser.parse_args()

op_fn = OPS[args.op]
op_name = args.op

def MODEL(a, b):
    return op_fn(a, b)

# Prepare inputs
if args.file:
    arr = np.fromfile(args.file, dtype=np.float32)
    if arr.size % 2 != 0:
        print('Input file must contain pairs of floats (a, b)')
        sys.exit(1)
    arr = arr.reshape(-1, 2)
    input_a = torch.from_numpy(arr[:, 0]).reshape(1, -1).to("cuda:0")
    input_b = torch.from_numpy(arr[:, 1]).reshape(1, -1).to("cuda:0")
else:
    input_a = torch.randn([1, args.size], dtype=torch.float32, device="cuda:0")
    input_b = torch.randn([1, args.size], dtype=torch.float32, device="cuda:0")

# Get baseline result from torch
torcheager = MODEL(input_a, input_b)

# Run first compile and execution
compiled_module = torch.compile(MODEL)
torchinductor = compiled_module(input_a, input_b)

# check that the results are the same
max_error = torch.max(torch.abs(torcheager - torchinductor)).item()
if torch.allclose(torcheager, torchinductor, atol=1e-06):
    print(f"\x1b[0;32;40m Torch {op_name} results and Diluvian results are the same: Max Error = {max_error:.8f} \033[0m")
else:
    print(f"Torch {op_name} results and Diluvian results differ: Max Error = {max_error:.8f}")

print(input_a)
print(input_b)
print(torcheager)
print(torchinductor)

# Save as raw binary (include op name in filenames)
eager_file = f'torcheager{op_name}.bin'
inductor_file = f'torchinductor{op_name}.bin'
torcheager.to("cpu").numpy().tofile(eager_file)
torchinductor.to("cpu").numpy().tofile(inductor_file)
print(f"Wrote {eager_file} and {inductor_file}")

# vim: et ts=4 sw=4
