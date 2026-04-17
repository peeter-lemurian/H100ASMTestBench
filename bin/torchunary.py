#! /usr/bin/env python3

# fmt: off
import torch
import argparse
import sys
import numpy as np

#import lemurian
#lemurian.lemurian_context().__enter__()

def roundf_away_from_zero(x):
    # Compute in float64 so values like 0.5-ulp are not rounded up by float32
    # before floor(); cast back to preserve expected output dtype.
    x64 = x.to(torch.float64)
    y64 = torch.sign(x64) * torch.floor(torch.abs(x64) + 0.5)
    return y64.to(x.dtype)

# Supported operations
OPS = {
    'expf': torch.exp,
    'exp2f': torch.exp2,
    'reciprocal': torch.reciprocal,
    'cosf': torch.cos,
    'sinf': torch.sin,
    'sqrtf': torch.sqrt,
    'rsqrtf': torch.rsqrt,
    'floorf': torch.floor,
    'truncf': torch.trunc,
    'ceilf': torch.ceil,
    'absf': torch.abs,
    'log2f': torch.log2,
    'logf': torch.log,
    'log10f': torch.log10,
    'erff': torch.erf,
    'erfcf': torch.erfc,
    'tanf': torch.tan,
    'tanhf': torch.tanh,
    'cbrtf': lambda x: torch.sign(x) * torch.abs(x).pow(1.0 / 3.0),
    'expm1f': torch.expm1,
    'log1pf': torch.log1p,
    'acoshf': torch.acosh,
    'asinhf': torch.asinh,
    'atanhf': torch.atanh,
    'acosf': torch.acos,
    'asinf': torch.asin,
    'atanf': torch.atan,
    # torch.round uses ties-to-even (banker's rounding), but C/HIP roundf is
    # half-away-from-zero, so we implement roundf semantics explicitly here.
    'roundf': roundf_away_from_zero,
    'nearbyintf': torch.round,
    'sinhf': torch.sinh,
    'coshf': torch.cosh,
    'exp10f': lambda x: torch.pow(10.0, x),
}

# Add optional ops only when the local torch build exposes them.
if hasattr(torch, 'logb'):
    OPS['logbf'] = torch.logb

if hasattr(torch, 'isnan'):
    OPS['isnanf'] = torch.isnan

if hasattr(torch, 'isinf'):
    OPS['isinff'] = torch.isinf

if hasattr(torch, 'isfinite'):
    OPS['isfinitef'] = torch.isfinite

if hasattr(torch, 'signbit'):
    OPS['signbitf'] = torch.signbit

if hasattr(torch, 'lgamma'):
    OPS['lgammaf'] = torch.lgamma

if hasattr(torch, 'special') and hasattr(torch.special, 'gamma'):
    OPS['tgammaf'] = torch.special.gamma

if hasattr(torch, 'special') and hasattr(torch.special, 'sinpi'):
    OPS['sinpif'] = torch.special.sinpi

if hasattr(torch, 'special') and hasattr(torch.special, 'cospi'):
    OPS['cospif'] = torch.special.cospi

if hasattr(torch, 'special') and hasattr(torch.special, 'tanpi'):
    OPS['tanpif'] = torch.special.tanpi

if hasattr(torch, 'frexp'):
    OPS['frexpf_mantissa'] = lambda x: torch.frexp(x)[0]
    OPS['frexpf_exponent'] = lambda x: torch.frexp(x)[1].to(torch.float32)

parser = argparse.ArgumentParser(description='Run a unary torch operation and save results.')
parser.add_argument('--op', choices=OPS.keys(), required=True,
                    help='Unary operation to compute (expf, cosf, sinf, sqrtf, rsqrtf, ...)')
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
if torcheager.dtype == torch.bool:
    mismatch = torch.logical_xor(torcheager, torchinductor)
    max_error = mismatch.float().max().item()
    same = torch.equal(torcheager, torchinductor)
else:
    max_error = torch.max(torch.abs(torcheager - torchinductor)).item()
    same = torch.allclose(torcheager, torchinductor, atol=1e-06)

if same:
    print(f"\x1b[0;32;40m Torch Eager {op_name} results and Torch Inductor results are the same: Max Error = {max_error:.8f} \033[0m")
else:
    print(f"Torch Eager {op_name} results and Torch Inductor results differ: Max Error = {max_error:.8f}")

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
