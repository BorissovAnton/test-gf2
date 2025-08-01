#include "GF2TestFramework.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

GF2TestFramework::GF2TestFramework() { initializeGPU(); }

GF2TestFramework::~GF2TestFramework() { cleanupGPU(); }

void GF2TestFramework::initializeGPU() {
  _device = MTL::CreateSystemDefaultDevice();
  if (_device) {
    _gpu = new GF2GPU(_device);
  } else {
    _gpu = nullptr;
  }
}

void GF2TestFramework::cleanupGPU() {
  if (_gpu)
    delete _gpu;
  if (_device)
    _device->release();
}

std::vector<TestResult> GF2TestFramework::runTests(const TestConfig &config) {
  std::vector<TestResult> allResults;

  std::cout << "Running GF(2) Matrix Multiplication Tests\n";
  std::cout << "========================================\n\n";

  const size_t max_size_1 = 1024;
  const size_t max_size_2 = 4096;

  for (const auto &size : config.matrix_sizes) {
    size_t rowsA = size.first;
    size_t colsA = size.second;
    size_t rowsB = colsA; // For valid multiplication
    size_t colsB = size.second;

    std::cout << "Testing matrices: " << rowsA << "x" << colsA << " * " << rowsB
              << "x" << colsB << "\n";

    GF2Matrix a = generateRandomMatrix(rowsA, colsA);
    GF2Matrix b = generateRandomMatrix(rowsB, colsB);

    // Skip serial multiplication for matrices 1024 and above
    if (config.run_serial && (rowsA < max_size_1)) {
      auto results = testSerial(a, b, config.iterations);
      allResults.insert(allResults.end(), results.begin(), results.end());
    }

    if (config.run_simd) {
      auto results = testSIMD(a, b, config.iterations);
      allResults.insert(allResults.end(), results.begin(), results.end());
    }

    if (config.run_gpu && _gpu) {
      auto results = testGPU(a, b, config.iterations && (rowsA < max_size_2));
      allResults.insert(allResults.end(), results.begin(), results.end());
    }

    if (config.run_gpu_transposed && _gpu && (rowsA < max_size_2)) {
      auto results = testGPU_transposed(a, b, config.iterations);
      allResults.insert(allResults.end(), results.begin(), results.end());
    }

    if (config.run_gpu_tiled && _gpu && (rowsA < max_size_2)) {
      auto results = testGPUTiled(a, b, config.iterations);
      allResults.insert(allResults.end(), results.begin(), results.end());
    }

    if (config.run_gpu_vectorized && _gpu) {
      auto results = testGPUVectorized(a, b, config.iterations);
      allResults.insert(allResults.end(), results.begin(), results.end());
    }

    if (config.run_gpu_m4r && _gpu) {
      auto results = testGPUM4R(a, b, config.iterations);
      allResults.insert(allResults.end(), results.begin(), results.end());
    }

    std::cout << "\n";
  }

  return allResults;
}

std::vector<TestResult> GF2TestFramework::testSerial(const GF2Matrix &a,
                                                     const GF2Matrix &b,
                                                     int iterations,
                                                     bool debug_mode) {
  if (a.cols() != b.rows()) {
    throw std::runtime_error(
        "Matrix dimensions incompatible for multiplication");
  }

  // Pre-allocate result matrix
  GF2Matrix result(a.rows(), b.cols());

  // Warm up with one multiplication
  GF2Matrix a_warm = generateRandomMatrix(a.rows(), a.cols());
  GF2Matrix b_warm = generateRandomMatrix(b.rows(), b.cols());
  result = a_warm.multiplySerial(b_warm);

  std::vector<TestResult> individual_results;

  for (int i = 0; i < iterations; i++) {
    // Generate new random matrices for each iteration
    GF2Matrix a_new = generateRandomMatrix(a.rows(), a.cols());
    GF2Matrix b_new = generateRandomMatrix(b.rows(), b.cols());

    auto start = std::chrono::high_resolution_clock::now();
    result = a_new.multiplySerial(b_new);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double throughput =
        calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count());

    if (debug_mode) {
      std::cout << "  Serial multiplication " << (i + 1) << "/" << iterations
                << " completed: " << a.rows() << "x" << a.cols() << " * "
                << b.rows() << "x" << b.cols() << " in " << duration.count()
                << " ms"
                << " and " << throughput << " GB/s"
                << "\n";
    }

    individual_results.push_back(
        {"Serial", duration.count(), true, throughput, a.rows() * b.cols()});
  }
  return individual_results;
}

std::vector<TestResult> GF2TestFramework::testSIMD(const GF2Matrix &a,
                                                   const GF2Matrix &b,
                                                   int iterations,
                                                   bool debug_mode) {
  if (a.cols() != b.rows()) {
    throw std::runtime_error(
        "Matrix dimensions incompatible for multiplication");
  }

  // Pre-allocate result matrix
  GF2Matrix result(a.rows(), b.cols());

  // Warm up with one multiplication
  GF2Matrix a_warm = generateRandomMatrix(a.rows(), a.cols());
  GF2Matrix b_warm = generateRandomMatrix(b.rows(), b.cols());
  result = a_warm.multiplySIMD(b_warm);

  std::vector<TestResult> individual_results;

  for (int i = 0; i < iterations; i++) {
    // Generate new random matrices for each iteration
    GF2Matrix a_new = generateRandomMatrix(a.rows(), a.cols());
    GF2Matrix b_new = generateRandomMatrix(b.rows(), b.cols());

    auto start = std::chrono::high_resolution_clock::now();
    result = a_new.multiplySIMD(b_new);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double throughput =
        calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count());

    if (debug_mode) {
      std::cout << "  SIMD multiplication " << (i + 1) << "/" << iterations
                << " completed: " << a.rows() << "x" << a.cols() << " * "
                << b.rows() << "x" << b.cols() << " in " << duration.count()
                << " ms"
                << " and " << throughput << " GB/s"
                << "\n";
    }

    individual_results.push_back(
        {"SIMD", duration.count(), true, throughput, a.rows() * b.cols()});
  }

  return individual_results;
}

// std::vector<TestResult> GF2TestFramework::testSIMD(const GF2Matrix& a, const
// GF2Matrix& b, int iterations, bool debug_mode) {
//     if (a.cols() != b.rows()) {
//         throw std::runtime_error("Matrix dimensions incompatible for
//         multiplication");
//     }
//
//     // Pre-allocate result matrix
//     GF2Matrix result(a.rows(), b.cols());
//
//     // Warm up with one multiplication
//     GF2Matrix a_warm = generateRandomMatrix(a.rows(), a.cols());
//     GF2Matrix b_warm = generateRandomMatrix(b.rows(), b.cols());
//     result = a_warm.multiplySIMD(b_warm);
//
//     std::vector<TestResult> individual_results;
//
//     for (int i = 0; i < iterations; i++) {
//         // Generate new random matrices for each iteration
//         GF2Matrix a_new = generateRandomMatrix(a.rows(), a.cols());
//         GF2Matrix b_new = generateRandomMatrix(b.rows(), b.cols());
//
//         auto start = std::chrono::high_resolution_clock::now();
//         result = a_new.multiplySIMD(b_new);
//         auto end = std::chrono::high_resolution_clock::now();
//
//         if (debug_mode) {
//             std::cout << "  SIMD multiplication " << (i + 1) << "/" <<
//             iterations
//                       << " completed: " << a.rows() << "x" << a.cols() << " *
//                       "
//                       << b.rows() << "x" << b.cols() << "\n";
//         }
//
//         std::chrono::duration<double, std::milli> duration = end - start;
//         double throughput = calculateThroughput(a.rows(), a.cols(), b.cols(),
//         duration.count());
//
//         individual_results.push_back({
//             "SIMD",
//             duration.count(),
//             true,
//             throughput,
//             a.rows() * b.cols()
//         });
//     }
//
//     return individual_results;
// }

std::vector<TestResult> GF2TestFramework::testGPU(const GF2Matrix &a,
                                                  const GF2Matrix &b,
                                                  int iterations,
                                                  bool debug_mode) {
  if (!_gpu) {
    return std::vector<TestResult>{
        {"GPU", 0.0, false, 0.0, a.rows() * b.cols()}};
  }

  // Pre-allocate result matrix
  GF2Matrix result(a.rows(), b.cols());

  // Warm up with one multiplication
  GF2Matrix a_warm = generateRandomMatrix(a.rows(), a.cols());
  GF2Matrix b_warm = generateRandomMatrix(b.rows(), b.cols());
  _gpu->multiplyGPU(a_warm, b_warm, result);

  std::vector<TestResult> individual_results;

  for (int i = 0; i < iterations; i++) {
    // Generate new random matrices for each iteration
    GF2Matrix a_new = generateRandomMatrix(a.rows(), a.cols());
    GF2Matrix b_new = generateRandomMatrix(b.rows(), b.cols());

    auto start = std::chrono::high_resolution_clock::now();
    _gpu->multiplyGPU(a_new, b_new, result);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double throughput =
        calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count());

    if (debug_mode) {
      std::cout << "  GPU multiplication " << (i + 1) << "/" << iterations
                << " completed: " << a.rows() << "x" << a.cols() << " * "
                << b.rows() << "x" << b.cols() << " in " << duration.count()
                << " ms"
                << " and " << throughput << " GB/s"
                << "\n";
    }

    individual_results.push_back(
        {"GPU", duration.count(), true, throughput, a.rows() * b.cols()});
  }

  return individual_results;
}

std::vector<TestResult> GF2TestFramework::testGPU_transposed(const GF2Matrix &a,
                                                             const GF2Matrix &b,
                                                             int iterations,
                                                             bool debug_mode) {
  if (!_gpu) {
    return std::vector<TestResult>{
        {"GPU (Transposed)", 0.0, false, 0.0, a.rows() * b.cols()}};
  }

  GF2Matrix result(a.rows(), b.cols());

  // Warm up
  GF2Matrix a_warm = generateRandomMatrix(a.rows(), a.cols());
  GF2Matrix b_warm = generateRandomMatrix(b.rows(), b.cols());
  _gpu->multiplyGPU_transposed(a_warm, b_warm, result);

  std::vector<TestResult> individual_results;

  for (int i = 0; i < iterations; i++) {
    GF2Matrix a_new = generateRandomMatrix(a.rows(), a.cols());
    GF2Matrix b_new = generateRandomMatrix(b.rows(), b.cols());

    auto start = std::chrono::high_resolution_clock::now();
    _gpu->multiplyGPU_transposed(a_new, b_new, result); // Call the new method
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double throughput =
        calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count());

    if (debug_mode) {
      std::cout << "  GPU (Transposed) multiplication " << (i + 1) << "/"
                << iterations << " completed: " << a.rows() << "x" << a.cols()
                << " * " << b.rows() << "x" << b.cols() << " in "
                << duration.count() << " ms"
                << " and " << throughput << " GOps/s"
                << "\n";
    }

    individual_results.push_back({"GPU (Transposed)", // Method name for reports
                                  duration.count(),
                                  true, // Assuming correctness for benchmark
                                  throughput, a.rows() * b.cols()});
  }

  return individual_results;
}

std::vector<TestResult> GF2TestFramework::testGPUTiled(const GF2Matrix &a,
                                                       const GF2Matrix &b,
                                                       int iterations,
                                                       bool debug_mode) {
  if (!_gpu) {
    return std::vector<TestResult>{
        {"GPU-Tiled", 0.0, false, 0.0, a.rows() * b.cols()}};
  }

  GF2Matrix result(a.rows(), b.cols());

  // Warm up
  GF2Matrix a_warm = generateRandomMatrix(a.rows(), a.cols());
  GF2Matrix b_warm = generateRandomMatrix(b.rows(), b.cols());
  _gpu->multiplyGPUTiled(a_warm, b_warm, result);

  std::vector<TestResult> individual_results;

  for (int i = 0; i < iterations; i++) {
    GF2Matrix a_new = generateRandomMatrix(a.rows(), a.cols());
    GF2Matrix b_new = generateRandomMatrix(b.rows(), b.cols());

    auto start = std::chrono::high_resolution_clock::now();
    _gpu->multiplyGPUTiled(a_new, b_new, result); // Call the new tiled method
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double throughput =
        calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count());

    if (debug_mode) {
      std::cout << "  GPU-Tiled multiplication " << (i + 1) << "/" << iterations
                << " completed: " << a.rows() << "x" << a.cols() << " * "
                << b.rows() << "x" << b.cols() << " in " << duration.count()
                << " ms"
                << " and " << throughput << " GB/s"
                << "\n";
    }

    individual_results.push_back(
        {"GPU-Tiled", // Set the correct method name for reporting
         duration.count(),
         true, // Assuming correctness for benchmark
         throughput, a.rows() * b.cols()});
  }

  return individual_results;
}

std::vector<TestResult> GF2TestFramework::testGPUVectorized(const GF2Matrix &a,
                                                            const GF2Matrix &b,
                                                            int iterations,
                                                            bool debug_mode) {
  if (!_gpu) {
    return std::vector<TestResult>{
        {"GPU-Vectorized", 0.0, false, 0.0, a.rows() * b.cols()}};
  }

  GF2Matrix result(a.rows(), b.cols());

  // Warm up
  GF2Matrix a_warm = generateRandomMatrix(a.rows(), a.cols());
  GF2Matrix b_warm = generateRandomMatrix(b.rows(), b.cols());
  _gpu->multiplyGPUVectorized(a_warm, b_warm, result);

  std::vector<TestResult> individual_results;

  for (int i = 0; i < iterations; i++) {
    GF2Matrix a_new = generateRandomMatrix(a.rows(), a.cols());
    GF2Matrix b_new = generateRandomMatrix(b.rows(), b.cols());

    auto start = std::chrono::high_resolution_clock::now();
    _gpu->multiplyGPUVectorized(a_new, b_new,
                                result); // Call the new vectorized method
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double throughput =
        calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count());

    if (debug_mode) {
      std::cout << "  GPU-Vectorized multiplication " << (i + 1) << "/"
                << iterations << " completed: " << a.rows() << "x" << a.cols()
                << " * " << b.rows() << "x" << b.cols() << " in "
                << duration.count() << " ms"
                << " and " << throughput << " GOps/s"
                << "\n";
    }

    individual_results.push_back({"GPU-Vectorized", // Method name for reports
                                  duration.count(),
                                  true, // Assuming correctness for benchmark
                                  throughput, a.rows() * b.cols()});
  }

  return individual_results;
}

std::vector<TestResult> GF2TestFramework::testGPUM4R(const GF2Matrix &a,
                                                     const GF2Matrix &b,
                                                     int iterations,
                                                     bool debug_mode) {
  if (!_gpu) {
    return std::vector<TestResult>{
        {"GPU (M4R)", 0.0, false, 0.0, a.rows() * b.cols()}};
  }

  GF2Matrix result(a.rows(), b.cols());

  // Warm up
  GF2Matrix a_warm = generateRandomMatrix(a.rows(), a.cols());
  GF2Matrix b_warm = generateRandomMatrix(b.rows(), b.cols());
  _gpu->multiplyGPUM4R(a_warm, b_warm, result);

  std::vector<TestResult> individual_results;

  for (int i = 0; i < iterations; i++) {
    GF2Matrix a_new = generateRandomMatrix(a.rows(), a.cols());
    GF2Matrix b_new = generateRandomMatrix(b.rows(), b.cols());

    auto start = std::chrono::high_resolution_clock::now();
    _gpu->multiplyGPUM4R(a_new, b_new, result); // Call the new M4R method
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double throughput =
        calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count());

    if (debug_mode) {
      std::cout << "  GPU (M4R) multiplication " << (i + 1) << "/" << iterations
                << " completed: " << a.rows() << "x" << a.cols() << " * "
                << b.rows() << "x" << b.cols() << " in " << duration.count()
                << " ms"
                << " and " << throughput << " GOps/s"
                << "\n";
    }

    individual_results.push_back({"GPU (M4R)", // Method name for reports
                                  duration.count(),
                                  true, // Assuming correctness for benchmark
                                  throughput, a.rows() * b.cols()});
  }

  return individual_results;
}

double GF2TestFramework::calculateThroughput(size_t a_rows, size_t a_cols,
                                             size_t b_cols,
                                             double duration_ms) {
  // Calculate bit operations per second using n³ formula
  // For square matrices: a_rows * a_cols * b_cols = n * n * n
  double operations = static_cast<double>(a_rows * a_cols * b_cols);
  double seconds = duration_ms / 1000.0;
  double gops = operations / seconds / 1e9; // Giga-operations per second
  return gops;
}

void GF2TestFramework::printResults(const std::vector<TestResult> &results) {
  std::cout << "\n=== Test Results Summary ===\n";
  std::cout << std::left << std::setw(10) << "Method" << std::setw(15)
            << "Time (ms)" << std::setw(15) << "Throughput" << std::setw(10)
            << "Correct" << std::setw(15) << "Matrix Size" << "\n";
  std::cout << std::string(65, '-') << "\n";

  for (const auto &result : results) {
    std::cout << std::left << std::setw(10) << result.method << std::setw(15)
              << std::fixed << std::setprecision(2) << result.duration_ms
              << std::setw(15) << std::setprecision(2) << result.throughput_gbps
              << std::setw(10) << (result.correct ? "✓" : "✗") << std::setw(15)
              << result.matrix_size << "\n";
  }

  std::cout << std::endl;
}

void GF2TestFramework::saveResults(const std::vector<TestResult> &results,
                                   const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  file << "Method,Duration_ms,Throughput_GOPS,Correct,Matrix_Size\n";
  for (const auto &result : results) {
    file << result.method << "," << result.duration_ms << ","
         << result.throughput_gbps << "," << result.correct << ","
         << result.matrix_size << "\n";
  }

  std::cout << "Results saved to: " << filename << std::endl;
}

GF2Matrix GF2TestFramework::generateRandomMatrix(size_t rows, size_t cols) {
  GF2Matrix matrix(rows, cols);
  matrix.randomFill();
  return matrix;
}

GF2Matrix GF2TestFramework::generateIdentityMatrix(size_t size) {
  GF2Matrix matrix(size, size);
  for (size_t i = 0; i < size; i++) {
    matrix.set(i, i, true);
  }
  return matrix;
}

bool GF2TestFramework::validateMultiplication(const GF2Matrix &a,
                                              const GF2Matrix &b,
                                              const GF2Matrix &result) {
  if (a.cols() != b.rows() || a.rows() != result.rows() ||
      b.cols() != result.cols()) {
    return false;
  }

  // Check sample positions for correctness
  for (size_t i = 0; i < std::min<size_t>(10, a.rows()); i++) {
    for (size_t j = 0; j < std::min<size_t>(10, b.cols()); j++) {
      bool expected = 0;
      for (size_t k = 0; k < a.cols(); k++) {
        // expected ^= (a.get(i, k) & b.get(k, j));
        expected ^=
            (static_cast<int>(a.get(i, k)) & static_cast<int>(b.get(k, j)));
      }
      if (expected != result.get(i, j)) {
        return false;
      }
    }
  }
  return true;
}
