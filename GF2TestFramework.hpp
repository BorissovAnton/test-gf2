#pragma once

#include "GF2Matrix.hpp"
#include "GF2GPU.hpp"
#include <chrono>
#include <vector>
#include <string>
#include <map>

struct TestResult {
    std::string method;
    double duration_ms;
    bool correct;
    double throughput_gbps; // Giga-bit operations per second
    size_t matrix_size;
};

struct TestConfig {
    std::vector<std::pair<size_t, size_t>> matrix_sizes;
    int iterations = 5;
    bool validate_results = true;
    bool run_serial = true;
    bool run_simd = true;
    bool run_gpu = true;
    bool run_gpu_transposed = true;
};

class GF2TestFramework {
public:
    GF2TestFramework();
    ~GF2TestFramework();
    
    // Run comprehensive test suite
    std::vector<TestResult> runTests(const TestConfig& config);
    
    // Individual test methods
    std::vector<TestResult> testSerial(const GF2Matrix& a, const GF2Matrix& b, int iterations, bool debug_mode = true);
    std::vector<TestResult> testSIMD(const GF2Matrix& a, const GF2Matrix& b, int iterations, bool debug_mode = true);
    std::vector<TestResult> testGPU(const GF2Matrix& a, const GF2Matrix& b, int iterations, bool debug_mode = true);
    std::vector<TestResult> testGPU_transposed(const GF2Matrix& a, const GF2Matrix& b, int iterations, bool debug_mode = true);

    
    // Performance reporting
    void printResults(const std::vector<TestResult>& results);
    void saveResults(const std::vector<TestResult>& results, const std::string& filename);
    
    // Matrix generation helpers
    static GF2Matrix generateRandomMatrix(size_t rows, size_t cols);
    static GF2Matrix generateIdentityMatrix(size_t size);
    static bool validateMultiplication(const GF2Matrix& a, const GF2Matrix& b, const GF2Matrix& result);
    
private:
    MTL::Device* _device;
    GF2GPU* _gpu;
    
    double calculateThroughput(size_t a_rows, size_t a_cols, size_t b_cols, double duration_ms);
    void initializeGPU();
    void cleanupGPU();
};

class BenchmarkSuite {
public:
    void runScalingBenchmark();
    void runThroughputBenchmark();
    void runAccuracyBenchmark();
    void generateReport();
};
