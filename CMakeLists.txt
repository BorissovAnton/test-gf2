# CMakeLists.txt for GF(2) Matrix Multiplication Test Suite

cmake_minimum_required(VERSION 3.16)
project(GF2MatrixTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP REQUIRED)

include_directories(metal-cpp)

# Check for Metal support on macOS
if(APPLE)
  find_library(METAL_LIBRARY Metal)
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(QUARTZCORE_LIBRARY QuartzCore)

  if(METAL_LIBRARY AND FOUNDATION_LIBRARY)
    set(METAL_SUPPORTED TRUE)
    message(STATUS "Metal support enabled")
  else()
    set(METAL_SUPPORTED FALSE)
    message(WARNING "Metal not found, GPU tests will be disabled")
  endif()
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

if(APPLE)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -framework Metal -framework Foundation -framework QuartzCore"
  )
endif()

# Add AVX2/AVX-512 support if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag(-mavx512f COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX2)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

if(COMPILER_SUPPORTS_AVX512)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
endif()

# Source files
set(SOURCES main.cpp GF2Matrix.cpp GF2GPU.cpp GF2TestFramework.cpp)

set(HEADERS GF2Matrix.hpp GF2GPU.hpp GF2TestFramework.hpp)

# Metal shader files
set(METAL_SOURCES gf2_multiply.metal)

# Create executable
add_executable(gf2_test ${SOURCES} ${HEADERS})

# Link libraries
if(APPLE AND METAL_SUPPORTED)
  target_link_libraries(gf2_test ${METAL_LIBRARY} ${FOUNDATION_LIBRARY}
                        ${QUARTZCORE_LIBRARY})
endif()

# Link OpenMP
if(OpenMP_CXX_FOUND)
  target_link_libraries(gf2_test OpenMP::OpenMP_CXX)
endif()

# Include directories
target_include_directories(gf2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                            ../metal-cpp)

# Copy Metal shaders to build directory
foreach(metal_file ${METAL_SOURCES})
  configure_file(${metal_file} ${CMAKE_CURRENT_BINARY_DIR}/${metal_file}
                 COPYONLY)
endforeach()

# Install target
install(TARGETS gf2_test DESTINATION bin)

# Create custom target for running tests
add_custom_target(
  run_gf2_tests
  COMMAND $<TARGET_FILE:gf2_test>
  DEPENDS gf2_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running GF(2) matrix tests")

# Install metal shaders
install(FILES ${METAL_SOURCES} DESTINATION bin)
