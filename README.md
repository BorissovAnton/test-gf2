# GF(2) Matrix Multiplication Test Suite

A comprehensive test suite for comparing serial, SIMD, and GPU implementations of GF(2) (Galois Field of two elements) matrix multiplication.

## Overview

This project provides three implementations of matrix multiplication over GF(2):

- **Serial**: Standard nested loop implementation
- **SIMD**: Vectorized implementation using AVX2/AVX-512
- **GPU**: Metal-accelerated implementation for Apple GPUs

## Features

- Comprehensive performance benchmarking
- Correctness validation against serial reference
- Configurable matrix sizes and iteration counts
- CSV output for further analysis
- Support for both small and large matrix operations

## Building

### Prerequisites

- macOS with Xcode command line tools
- CMake 3.16 or higher
- Metal support (for GPU tests)

### Build Instructions

```bash
mkdir build && cd build
cmake ../test-gf2
make -j$(sysctl -n hw.ncpu)
```

### Running Tests

```bash
./gf2_test                    # Run with default settings
./gf2_test 256 10             # Test 256x256 matrices with 10 iterations
./gf2_test 1024 5             # Test 1024x1024 matrices with 5 iterations
```

## Architecture

### Core Components

1. **GF2Matrix**: Main matrix class with bit-packed storage
2. **GF2GPU**: Metal-based GPU acceleration
3. **GF2TestFramework**: Comprehensive testing and benchmarking
4. **Performance profiling**: Accurate timing and throughput calculation

### File Structure

```
test-gf2/
├── GF2Matrix.hpp/.cpp      # Matrix class implementation
├── GF2GPU.hpp/.cpp         # GPU acceleration
├── GF2TestFramework.hpp/.cpp # Testing framework
├── GF2MatrixSIMD.cpp       # SIMD optimizations
├── gf2_multiply.metal      # Metal shaders
├── main.cpp               # Main test runner
├── CMakeLists.txt         # Build configuration
├── README.md             # This file
└── gf2_test_results.csv  # Generated results
```

## Performance Notes

- **Serial**: Baseline performance, good for validation
- **SIMD**: Significant speedup for medium/large matrices
- **GPU**: Best performance for very large matrices (1K+ dimensions)

## Testing

### Default Test Suite

Tests matrix sizes: 64×64, 128×128, 256×256, 512×512, 1024×1024

### Validation Tests

- Identity matrix multiplication
- Small matrix accuracy verification
- Cross-implementation consistency checks

## Output

Results are saved to `gf2_test_results.csv` with columns:

- Method: Implementation type (Serial/SIMD/GPU)
- Duration_ms: Average execution time in milliseconds
- Throughput_GOPS: Giga-operations per second
- Correct: Validation status
- Matrix_Size: Matrix dimensions

## Usage Example

```cpp
#include "GF2TestFramework.hpp"

int main() {
    GF2TestFramework framework;

    TestConfig config;
    config.matrix_sizes = {{512, 512}};
    config.iterations = 10;

    auto results = framework.runTests(config);
    framework.printResults(results);

    return 0;
}
```
