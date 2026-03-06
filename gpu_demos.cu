// gpu_demos.cu - Misc CUDA demonstration program
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <getopt.h>
#include <stdexcept>
#include <string>

// ============================================================================
// Demo 1: Lane-based argument shuffling and DMA-style operations
// ============================================================================

extern "C" __device__ void dmaDropFunc3(uint64_t vargs) {
  // COLLECTION PHASE: Each thread uses __shfl_sync to collect values
  // from specific lanes. After all shuffles, EVERY thread has ALL values.

  // Extract output pointer from lane 0
  // BEFORE: only lane 0 has this value in its vargs
  // AFTER:  ALL 32 lanes have this value in ptrOut
  uint64_t ptrOut = __shfl_sync(0xffffffff, vargs, 0);

  // Extract input pointer from lane 1
  uint64_t ptrIn = __shfl_sync(0xffffffff, vargs, 1);

  // Convert to typed pointers (now all threads have both pointers)
  float *tileOutput = reinterpret_cast<float *>(ptrOut);
  float *tileInput = reinterpret_cast<float *>(ptrIn);

  // Extract scalar values from lanes 8, 9, 10, 11
  uint32_t scalarInput1 = __shfl_sync(0xffffffff, (uint32_t)vargs, 8);
  uint32_t scalarInput2 = __shfl_sync(0xffffffff, (uint32_t)vargs, 9);
  uint32_t scalarInput3 = __shfl_sync(0xffffffff, (uint32_t)vargs, 10);
  uint32_t scalarInput4 = __shfl_sync(0xffffffff, (uint32_t)vargs, 11);

  // At this point, EVERY thread has all the values

  // Compute offset using the scalar values (all threads compute same value)
  uint32_t offset =
      scalarInput1 + 4 * (scalarInput2 + 4 * (scalarInput3 + 8 * scalarInput4));
  offset *= 32;

  // Now do actual work - each thread processes one element
  int lane = threadIdx.x % 32;

  if (lane < 32 && tileInput && tileOutput) {
    tileOutput[offset + lane] = tileInput[offset + lane] * 2.0f;
  }
}

__global__ void dmaKernel(float *dInput, float *dOutput, uint32_t s1,
                          uint32_t s2, uint32_t s3, uint32_t s4) {
  // DISTRIBUTION PHASE: Each lane sets its vargs to a specific value
  int lane = threadIdx.x % 32;
  uint64_t vargs = 0;

  if (lane == 0) {
    vargs = reinterpret_cast<uint64_t>(dOutput);
  } else if (lane == 1) {
    vargs = reinterpret_cast<uint64_t>(dInput);
  } else if (lane == 8) {
    vargs = s1;
  } else if (lane == 9) {
    vargs = s2;
  } else if (lane == 10) {
    vargs = s3;
  } else if (lane == 11) {
    vargs = s4;
  }

  // Call the function which will shuffle-gather all values
  dmaDropFunc3(vargs);
}

// ============================================================================
// Demo 2: Query device properties
// ============================================================================

void queryOneDevice(int dev) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);

  printf("Device %d: %s\n"
         "  Compute Capability: %d.%d\n"
         "  Multiprocessors (SMs): %d\n"
         "  Registers per SM: %d\n"
         "  Registers per block (max): %d\n"
         "  Total registers: %d (%.2f MB)\n"
         "  Warp size: %d\n"
         "  Max threads per block: %d\n"
         "  Max threads per SM: %d\n"
         "  Shared memory per block: %zu bytes (%.2f KB)\n"
         "  Total global memory: %.2f GB\n"
         //"  Memory clock rate: %.2f GHz\n"
         "  Memory bus width: %d-bit\n"
         "  L2 cache size: %.2f MB\n"
         "\n",
         dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         prop.regsPerMultiprocessor, prop.regsPerBlock,
         prop.multiProcessorCount * prop.regsPerMultiprocessor,
         (prop.multiProcessorCount * prop.regsPerMultiprocessor * 4) /
             (1024.0 * 1024.0),
         prop.warpSize, prop.maxThreadsPerBlock,
         prop.maxThreadsPerMultiProcessor, prop.sharedMemPerBlock,
         prop.sharedMemPerBlock / 1024.0,
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
         // prop.memoryClockRate / 1e6,
         prop.memoryBusWidth, prop.l2CacheSize / (1024.0 * 1024.0));
}

void queryDeviceProperties(int device) {
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
    return;
  }

  printf("Found %d CUDA device(s)\n\n", deviceCount);

  if ((device >= 0) && device < deviceCount) {
    queryOneDevice(device);
  } else {
    for (int dev = 0; dev < deviceCount; dev++) {
      queryOneDevice(dev);
    }
  }
}

// ============================================================================
// Demo 3: Simple vector addition
// ============================================================================

__global__ void vectorAddKernel(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void demoVectorAdd(int n) {
  printf("Running vector addition demo with %d elements\n", n);

  size_t bytes = n * sizeof(float);

  float *hA = new float[n];
  float *hB = new float[n];
  float *hC = new float[n];

  for (int i = 0; i < n; i++) {
    hA[i] = static_cast<float>(i);
    hB[i] = static_cast<float>(i * 2);
  }

  float *dA, *dB, *dC;
  cudaMalloc(&dA, bytes);
  cudaMalloc(&dB, bytes);
  cudaMalloc(&dC, bytes);

  cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  vectorAddKernel<<<gridSize, blockSize>>>(dA, dB, dC, n);
#if 0 // that's like:
  void *args[] = {&dA, &dB, &dC, &n};
  
  cudaLaunchKernel(
      (void*)vectorAddKernel,    // Kernel function
      gridSize,                   // Grid dimensions
      blockSize,                  // Block dimensions
      args,                       // Kernel arguments (array of pointers)
      0,                          // Shared memory size
      0                           // Stream (0 = default stream)
  );
#endif

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

  // Verify first few elements
  bool correct = true;
  for (int i = 0; i < (n < 10 ? n : 10); i++) {
    float expected = hA[i] + hB[i];
    if (hC[i] != expected) {
      correct = false;
      printf("Error at index %d: expected %.1f, got %.1f\n", i, expected,
             hC[i]);
    }
  }

  if (correct) {
    printf("Vector addition: PASSED ✓\n"
           "Sample results: %.1f + %.1f = %.1f, %.1f + %.1f = %.1f\n",
           hA[0], hB[0], hC[0], hA[1], hB[1], hC[1]);
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  delete[] hA;
  delete[] hB;
  delete[] hC;
}

// ============================================================================
// Demo 4: Lane shuffle DMA demo
// ============================================================================

void demoLaneShuffle() {
  printf("Running lane shuffle DMA demo\n");

  const int N = 32768;
  const size_t bytes = N * sizeof(float);

  float *hInput = new float[N];
  float *hOutput = new float[N];

  for (int i = 0; i < N; i++) {
    hInput[i] = static_cast<float>(i);
    hOutput[i] = 0.0f;
  }

  float *dInput, *dOutput;
  cudaMalloc(&dInput, bytes);
  cudaMalloc(&dOutput, bytes);

  cudaMemcpy(dInput, hInput, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dOutput, hOutput, bytes, cudaMemcpyHostToDevice);

  uint32_t s1 = 1, s2 = 2, s3 = 3, s4 = 0;

  dmaKernel<<<1, 32>>>(dInput, dOutput, s1, s2, s3, s4);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  cudaMemcpy(hOutput, dOutput, bytes, cudaMemcpyDeviceToHost);

  int offset = (s1 + 4 * (s2 + 4 * (s3 + 8 * s4))) * 32;

  printf("Computed offset: %d\n"
         "Processed elements [%d..%d]\n",
         offset, offset, offset + 31);

  bool correct = true;
  for (int i = 0; i < 10 && (offset + i) < N; i++) {
    float expected = hInput[offset + i] * 2.0f;
    if (hOutput[offset + i] != expected) {
      correct = false;
      printf("Error at %d: expected %.1f, got %.1f\n", offset + i, expected,
             hOutput[offset + i]);
    }
  }

  if (correct) {
    printf("Lane shuffle DMA: PASSED ✓\n"
           "Sample: input[%d]=%.1f → output[%d]=%.1f (×2)\n",
           offset, hInput[offset], offset, hOutput[offset]);
  }

  cudaFree(dInput);
  cudaFree(dOutput);
  delete[] hInput;
  delete[] hOutput;
}

// ============================================================================
// Demo 5: Occupancy calculation
// ============================================================================

void demoOccupancy() {
  printf("Running occupancy calculation demo\n\n");

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
  int regsPerSM = prop.regsPerMultiprocessor;

  printf("Device: %s\n"
         "Max threads per SM: %d\n"
         "Registers per SM: %d\n\n",
         prop.name, maxThreadsPerSM, regsPerSM);

  int threadsPerBlock = 256;

  printf("Occupancy for %d threads per block:\n"
         "%-15s %-12s %-12s %-12s %-12s\n"
         "%-15s %-12s %-12s %-12s %-12s\n",
         threadsPerBlock, "Regs/Thread", "Blocks/SM", "Threads/SM", "Occupancy",
         "Wasted Regs", "---------------", "------------", "------------",
         "------------", "------------");

  int regCounts[] = {8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 255};

  for (int regsPerThread : regCounts) {
    int regsPerBlock = threadsPerBlock * regsPerThread;
    int blocksPerSM = regsPerSM / regsPerBlock;
    if (blocksPerSM > prop.maxBlocksPerMultiProcessor) {
      blocksPerSM = prop.maxBlocksPerMultiProcessor;
    }
    int threadsPerSM = blocksPerSM * threadsPerBlock;
    float occupancy = (100.0f * threadsPerSM) / maxThreadsPerSM;
    int wastedRegs = regsPerSM - (blocksPerSM * regsPerBlock);

    printf("%-15d %-12d %-12d %-11.1f%% %-12d\n", regsPerThread, blocksPerSM,
           threadsPerSM, occupancy, wastedRegs);
  }
}

void printUsage(const char *programName) {
  printf("Usage: %s [OPTIONS]\n\n"
         "CUDA GPU demonstration program with multiple demos.\n\n"
         "Options:\n"
         "  --help              Show this help message\n"
         "  --query [--dev N]   Query and display device properties\n"
         "  --vector-add [N]    Run vector addition demo (default N=1000000)\n"
         "  --lane-shuffle      Run lane shuffle DMA demo\n"
         "  --occupancy         Run occupancy calculation demo\n"
         "  --all               Run all demos\n\n"
         "Examples:\n"
         "  %s --query                    # Query device properties\n"
         "  %s --vector-add 100000        # Vector add with 100k elements\n"
         "  %s --lane-shuffle             # Run lane shuffle demo\n"
         "  %s --all                      # Run all demos\n"
         "  %s -q -v -l                   # Run query, vector add, and lane "
         "shuffle\n\n",
         programName, programName, programName, programName, programName,
         programName);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    printUsage(argv[0]);
    return 0;
  }

  bool runQuery = false;
  bool runVectorAdd = false;
  bool runLaneShuffle = false;
  bool runOccupancy = false;
  bool runAll = false;
  int vectorAddSize = 1000000;
  int deviceNumber = -1;

  enum class Options : int {
    help,
    query,
    vadd,
    laneshuffle,
    occupancy,
    all,
    device
  };

  static struct option longOptions[] = {
      {"help", no_argument, 0, (int)Options::help},
      {"query", no_argument, 0, (int)Options::query},
      {"device", required_argument, 0, (int)Options::device},
      {"vector-add", optional_argument, 0, (int)Options::vadd},
      {"lane-shuffle", no_argument, 0, (int)Options::laneshuffle},
      {"occupancy", no_argument, 0, (int)Options::occupancy},
      {"all", no_argument, 0, (int)Options::all},
      {0, 0, 0, 0}};

  int opt;
  int optionIndex = 0;

  try {
    while ((opt = getopt_long(argc, argv, "h", longOptions, &optionIndex)) !=
           -1) {
      switch ((Options)opt) {
      case Options::help:
        printUsage(argv[0]);
        return 0;

      case Options::query:
        runQuery = true;
        break;

      case Options::device:
        runQuery = true;
        deviceNumber = std::stoi(optarg);
        break;

      case Options::vadd:
        runVectorAdd = true;
        if (optarg) {
          vectorAddSize = std::stoi(optarg);
          if (vectorAddSize <= 0) {
            fprintf(stderr, "Error: vector size must be positive\n");
            return 1;
          }
        }
        break;

      case Options::laneshuffle:
        runLaneShuffle = true;
        break;

      case Options::occupancy:
        runOccupancy = true;
        break;

      case Options::all:
        runAll = true;
        break;

      case (Options)'?':
      default:
        fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
        return 1;
      }
    }
  } catch (const std::invalid_argument &e) {
    fprintf(stderr, "Error: invalid number '%s'\n", optarg);
    return 1;
  } catch (const std::out_of_range &e) {
    fprintf(stderr, "Error: number '%s' out of range\n", optarg);
    return 1;
  }

  // Handle --all flag
  if (runAll) {
    runQuery = true;
    runVectorAdd = true;
    runLaneShuffle = true;
    runOccupancy = true;
  }

  // Check if CUDA is available
  int deviceCount;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  if (error != cudaSuccess || deviceCount == 0) {
    fprintf(stderr, "Error: No CUDA devices found\n");
    return 1;
  }

  // Run selected demos
  bool first = true;

  if (runQuery) {
    if (!first)
      printf("\n"
             "========================================\n\n");
    queryDeviceProperties(deviceNumber);
    first = false;
  }

  if (runVectorAdd) {
    if (!first)
      printf("\n"
             "========================================\n\n");
    demoVectorAdd(vectorAddSize);
    first = false;
  }

  if (runLaneShuffle) {
    if (!first)
      printf("\n"
             "========================================\n\n");
    demoLaneShuffle();
    first = false;
  }

  if (runOccupancy) {
    if (!first)
      printf("\n"
             "========================================\n\n");
    demoOccupancy();
    first = false;
  }

  return 0;
}

// vim: et ts=2 sw=2
