# CMakeLists.txt for GF(2) Matrix Multiplication Test Suite

cmake_minimum_required(VERSION 3.16)
project(GF2MatrixTest LANGUAGES CXX OBJCXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ and Objective-C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP REQUIRED)

# Check for Metal support on macOS
if(APPLE)
  set(METAL_SUPPORTED TRUE)
  message(STATUS "Metal support enabled")
else()
  set(METAL_SUPPORTED FALSE)
  message(WARNING "Metal not found, GPU tests will be disabled")
endif()

# Compiler flags
set(COMMON_CXX_FLAGS "-Wall -Wextra -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} ${COMMON_CXX_FLAGS}")

# Add AVX2/AVX-512 support if available (for x86_64)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag(-mavx512f COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX2)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -mavx2")
endif()

if(COMPILER_SUPPORTS_AVX512)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -mavx512f")
endif()

# Source files
set(SOURCES main.cpp GF2Matrix.cpp GF2GPU.cpp GF2TestFramework.cpp)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  list(APPEND SOURCES GF2MatrixSIMD_x86.cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  list(APPEND SOURCES GF2MatrixSIMD_arm.cpp)
endif()
set(HEADERS GF2Matrix.hpp GF2GPU.hpp GF2TestFramework.hpp)

# Create executable
add_executable(gf2_test ${SOURCES} ${HEADERS})

# Set language to Objective-C++ for files that include Metal/Foundation headers
if(APPLE AND METAL_SUPPORTED)
  set_source_files_properties(main.cpp GF2GPU.cpp GF2TestFramework.cpp
                              PROPERTIES LANGUAGE OBJCXX)
endif()

# --- FIX: COMPILE ALL METAL SHADERS INTO A SINGLE LIBRARY ---
if(APPLE AND METAL_SUPPORTED)
  # This command compiles ALL .metal files into a SINGLE default.metallib file.
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    # FIX: List all input files first, then a single -o for the output.
    COMMAND
      xcrun -sdk macosx metal ${CMAKE_CURRENT_SOURCE_DIR}/gf2_multiply.metal
      ${CMAKE_CURRENT_SOURCE_DIR}/gf2_multiply_transposed.metal
      ${CMAKE_CURRENT_SOURCE_DIR}/gf2_multiply_tiled.metal
      ${CMAKE_CURRENT_SOURCE_DIR}/gf2_multiply_vectorized.metal -o
      ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    # FIX: The dependency list must include all source files.
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gf2_multiply.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/gf2_multiply_transposed.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/gf2_multiply_tiled.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/gf2_multiply_vectorized.metal
    COMMENT "Compiling all Metal shaders into default.metallib")

  add_custom_target(MetalLibrary
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

  add_dependencies(gf2_test MetalLibrary)
endif()
# --- END FIX ---

# Link libraries
if(APPLE AND METAL_SUPPORTED)
  target_link_libraries(
    gf2_test PRIVATE "-framework Metal" "-framework Foundation"
                     "-framework QuartzCore")
endif()

# Link OpenMP
if(OpenMP_CXX_FOUND)
  target_link_libraries(gf2_test PRIVATE OpenMP::OpenMP_CXX)
endif()

# Include directories
target_include_directories(gf2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                            metal-cpp)

# Create custom target for running tests
add_custom_target(
  run_gf2_tests
  COMMAND $<TARGET_FILE:gf2_test>
  DEPENDS gf2_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running GF(2) matrix tests")

# Install target
install(TARGETS gf2_test DESTINATION bin)

# Install the compiled metal library, not the source
if(APPLE AND METAL_SUPPORTED)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/default.metallib DESTINATION bin)
endif()
