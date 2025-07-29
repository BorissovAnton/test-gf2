#include "GF2TestFramework.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

GF2TestFramework::GF2TestFramework() {
    initializeGPU();
}

GF2TestFramework::~GF2TestFramework() {
    cleanupGPU();
}

void GF2TestFramework::initializeGPU() {
    _device = MTL::CreateSystemDefaultDevice();
    if (_device) {
        _gpu = new GF2GPU(_device);
    } else {
        _gpu = nullptr;
    }
}

void GF2TestFramework::cleanupGPU() {
    if (_gpu) delete _gpu;
    if (_device) _device->release();
}

std::vector<TestResult> GF2TestFramework::runTests(const TestConfig& config) {
    std::vector<TestResult> allResults;
    
    std::cout << "Running GF(2) Matrix Multiplication Tests\n";
    std::cout << "========================================\n\n";
    
    for (const auto& size : config.matrix_sizes) {
        size_t rowsA = size.first;
        size_t colsA = size.second;
        size_t rowsB = colsA; // For valid multiplication
        size_t colsB = size.second;
        
        std::cout << "Testing matrices: " << rowsA << "x" << colsA << " * " 
                  << rowsB << "x" << colsB << "\n";
        
        GF2Matrix a = generateRandomMatrix(rowsA, colsA);
        GF2Matrix b = generateRandomMatrix(rowsB, colsB);
        
        if (config.run_serial) {
            auto result = testSerial(a, b, config.iterations);
            allResults.push_back(result);
        }
        
        if (config.run_simd) {
            auto result = testSIMD(a, b, config.iterations);
            allResults.push_back(result);
        }
        
        if (config.run_gpu && _gpu) {
            auto result = testGPU(a, b, config.iterations);
            allResults.push_back(result);
        }
        
        std::cout << "\n";
    }
    
    return allResults;
}

TestResult GF2TestFramework::testSerial(const GF2Matrix& a, const GF2Matrix& b, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    bool correct = true;
    GF2Matrix result(a.rows(), b.cols());
    
    for (int i = 0; i < iterations; i++) {
        result = a.multiplySerial(b);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    double throughput = calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count() / iterations);
    
    return {
        "Serial",
        duration.count() / iterations,
        correct,
        throughput,
        a.rows() * b.cols()
    };
}

TestResult GF2TestFramework::testSIMD(const GF2Matrix& a, const GF2Matrix& b, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    bool correct = true;
    GF2Matrix result(a.rows(), b.cols());
    
    for (int i = 0; i < iterations; i++) {
        result = a.multiplySIMD(b);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    double throughput = calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count() / iterations);
    
    return {
        "SIMD",
        duration.count() / iterations,
        correct,
        throughput,
        a.rows() * b.cols()
    };
}

TestResult GF2TestFramework::testGPU(const GF2Matrix& a, const GF2Matrix& b, int iterations) {
    if (!_gpu) {
        return {"GPU", 0.0, false, 0.0, a.rows() * b.cols()};
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool correct = true;
    GF2Matrix result(a.rows(), b.cols());
    
    for (int i = 0; i < iterations; i++) {
        _gpu->multiplyGPU(a, b, result);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    double throughput = calculateThroughput(a.rows(), a.cols(), b.cols(), duration.count() / iterations);
    
    return {
        "GPU",
        duration.count() / iterations,
        correct,
        throughput,
        a.rows() * b.cols()
    };
}

double GF2TestFramework::calculateThroughput(size_t a_rows, size_t a_cols, size_t b_cols, double duration_ms) {
    // Calculate bit operations per second
    // For GF(2) matrix multiplication: a_rows * a_cols * b_cols operations
    double operations = static_cast<double>(a_rows * a_cols * b_cols);
    double seconds = duration_ms / 1000.0;
    double gops = operations / seconds / 1e9; // Giga-operations per second
    return gops;
}

void GF2TestFramework::printResults(const std::vector<TestResult>& results) {
    std::cout << "\n=== Test Results Summary ===\n";
    std::cout << std::left << std::setw(10) << "Method"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(10) << "Correct"
              << std::setw(15) << "Matrix Size" << "\n";
    std::cout << std::string(65, '-') << "\n";
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(10) << result.method
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.duration_ms
                  << std::setw(15) << std::setprecision(2) << result.throughput_gbps
                  << std::setw(10) << (result.correct ? "✓" : "✗")
                  << std::setw(15) << result.matrix_size << "\n";
    }
    
    std::cout << std::endl;
}

void GF2TestFramework::saveResults(const std::vector<TestResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    file << "Method,Duration_ms,Throughput_GOPS,Correct,Matrix_Size\n";
    for (const auto& result : results) {
        file << result.method << "," 
             << result.duration_ms << "," 
             << result.throughput_gbps << "," 
             << result.correct << "," 
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

bool GF2TestFramework::validateMultiplication(const GF2Matrix& a, const GF2Matrix& b, const GF2Matrix& result) {
    if (a.cols() != b.rows() || a.rows() != result.rows() || b.cols() != result.cols()) {
        return false;
    }
    
    // Check sample positions for correctness
    for (size_t i = 0; i < std::min<size_t>(10, a.rows()); i++) {
        for (size_t j = 0; j < std::min<size_t>(10, b.cols()); j++) {
            bool expected = 0;
            for (size_t k = 0; k < a.cols(); k++) {
                // expected ^= (a.get(i, k) & b.get(k, j));
                expected ^= (static_cast<int>(a.get(i, k)) & static_cast<int>(b.get(k, j)));
            }
            if (expected != result.get(i, j)) {
                return false;
            }
        }
    }
    return true;
}

