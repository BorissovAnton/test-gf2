#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "GF2TestFramework.hpp"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  std::cout << "GF(2) Matrix Multiplication Performance Test Suite\n";
  std::cout << "================================================\n\n";

  // Create test configuration
  TestConfig config;

  // Set test matrix sizes (adjust based on your hardware)
  config.matrix_sizes = {{64, 64},     {128, 128},   {256, 256},
                         {512, 512},   {1024, 1024}, {2048, 2048},
                         {4096, 4096}, {8192, 8192}};

  // Parse command line arguments
  if (argc > 1) {
    // int it = std::stoi(argv[1]);
    // config.matrix_sizes = {{size, size}};
    config.iterations = std::stoi(argv[1]);
  }

  // // Parse command line arguments
  // for (int i = 1; i < argc; ++i) {
  //     std::string arg = argv[i];
  //     if (arg == "--iterations" && i + 1 < argc) {
  //         config.iterations = std::stoi(argv[i + 1]);
  //         ++i; // Skip the next argument
  //     } else if (arg.find("--") != 0) {  // Not a flag, likely a number
  //         try {
  //             int size = std::stoi(arg);
  //             config.matrix_sizes = {{size, size}};
  //         } catch (const std::invalid_argument&) {
  //             // Ignore non-numeric arguments that aren't flags
  //         }
  //     }
  // }

  std::cout << "Configuration:\n";
  std::cout << "- Matrix sizes: ";
  for (const auto &size : config.matrix_sizes) {
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

    // Display progress summary
    std::cout << "\n=== Processing Summary ===\n";
    std::cout << "Total individual results collected: " << results.size() << "\n";
    
    // Group by method and matrix size
    std::map<std::string, std::map<size_t, int>> counts;
    for (const auto& result : results) {
        counts[result.method][result.matrix_size]++;
    }
    
    for (const auto& [method, size_counts] : counts) {
        std::cout << method << " results: ";
        for (const auto& [size, count] : size_counts) {
            std::cout << count << "x" << size << " ";
        }
        std::cout << "\n";
    }

    // Additional validation tests
    std::cout << "\n=== Validation Tests ===\n";

    // Test 1: Identity matrix multiplication
    std::cout << "Testing identity matrix multiplication...\n";
    GF2Matrix identity = GF2TestFramework::generateIdentityMatrix(128);
    GF2Matrix random = GF2TestFramework::generateRandomMatrix(128, 128);

    auto serial_result = identity.multiplySerial(random);
    auto simd_result = identity.multiplySIMD(random);

    bool identity_test = serial_result == simd_result;
    std::cout << "Identity test: " << (identity_test ? "PASSED" : "FAILED")
              << "\n";

    // Test 2: Small matrix accuracy
    std::cout << "Testing small matrix accuracy...\n";
    GF2Matrix small_a(32, 32);
    GF2Matrix small_b(32, 32);
    small_a.randomFill();
    small_b.randomFill();

    auto small_serial = small_a.multiplySerial(small_b);
    auto small_simd = small_a.multiplySIMD(small_b);

    bool small_test = small_serial == small_simd;
    std::cout << "Small matrix test: " << (small_test ? "PASSED" : "FAILED")
              << "\n";

    std::cout << "\n=== Test Suite Complete ===\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
