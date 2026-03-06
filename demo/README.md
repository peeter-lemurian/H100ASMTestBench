# Show help
./gpu_demos --help

# Query device properties
./gpu_demos --query

# Run vector addition with default size
./gpu_demos --vector-add

# Run vector addition with custom size
./gpu_demos --vector-add 500000

# Run lane shuffle demo
./gpu_demos --lane-shuffle

# Run occupancy calculation
./gpu_demos --occupancy

# Run all demos
./gpu_demos --all

# Combine multiple demos
./gpu_demos -q -v -l -o
```

**Example output:**
```
$ ./gpu_demos --help
Usage: ./gpu_demos [OPTIONS]

CUDA GPU demonstration program with multiple demos.

Options:
  -h, --help              Show this help message
  -q, --query             Query and display device properties
  -v, --vector-add [N]    Run vector addition demo (default N=1000000)
  -l, --lane-shuffle      Run lane shuffle DMA demo
  -o, --occupancy         Run occupancy calculation demo
  -a, --all               Run all demos

$ ./gpu_demos --query
Found 1 CUDA device(s)

Device 0: NVIDIA GeForce RTX 3080
  Compute Capability: 8.6
  Multiprocessors (SMs): 68
  ...

$ ./gpu_demos --vector-add 1000
Running vector addition demo with 1000 elements
Vector addition: PASSED ✓
Sample results: 0.0 + 0.0 = 0.0, 1.0 + 2.0 = 3.0
