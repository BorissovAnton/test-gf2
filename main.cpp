#include "GF2TestFramework.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "GF(2) Matrix Multiplication Performance Test Suite\n";
    std::cout << "================================================\n\n";
    
    // Create test configuration
    TestConfig config;
    
    // Set test matrix sizes (adjust based on your hardware)
    config.matrix_sizes = {
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024}
    };
    
    // Parse command line arguments
    if (argc > 1) {
        int size = std::stoi(argv[1]);
        config.matrix_sizes = {{size, size}};
    }
    
    if (argc > 2) {
        config.iterations = std::stoi(argv[2]);
    }
    
    std::cout << "Configuration:\n";
    std::cout << "- Matrix sizes: ";
    for (const auto& size : config.matrix_sizes) {
        std::cout << "(" << size.first << "x" << size.second << ") ";
    }
    std::cout << "\n";
    std::cout << "- Iterations per test: " << config.iterations << "\n\n";
    
    try {
        GF2TestFramework framework;
        
        // Run comprehensive tests
        auto results = framework.runTests(config);
        
        // Print and save results
        framework.printResults(results);
        framework.saveResults(results, "gf2_test_results.csv");
        
        // Additional validation tests
        std::cout << "\n=== Validation Tests ===\n";
        
        // Test 1: Identity matrix multiplication
        std::cout << "Testing identity matrix multiplication...\n";
        GF2Matrix identity = GF2TestFramework::generateIdentityMatrix(128);
        GF2Matrix random = GF2TestFramework::generateRandomMatrix(128, 128);
        
        auto serial_result = identity.multiplySerial(random);
        auto simd_result = identity.multiplySIMD(random);
        
        bool identity_test = serial_result == simd_result;
        std::cout << "Identity test: " << (identity_test ? "PASSED" : "FAILED") << "\n";
        
        // Test 2: Small matrix accuracy
        std::cout << "Testing small matrix accuracy...\n";
        GF2Matrix small_a(32, 32);
        GF2Matrix small_b(32, 32);
        small_a.randomFill();
        small_b.randomFill();
        
        auto small_serial = small_a.multiplySerial(small_b);
        auto small_simd = small_a.multiplySIMD(small_b);
        
        bool small_test = small_serial == small_simd;
        std::cout << "Small matrix test: " << (small_test ? "PASSED" : "FAILED") << "\n";
        
        std::cout << "\n=== Test Suite Complete ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

