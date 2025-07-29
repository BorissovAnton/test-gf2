#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "GF2Matrix.hpp"
#include <vector>

class GF2GPU {
public:
    GF2GPU(MTL::Device* device);
    ~GF2GPU();
    
    // GPU-accelerated matrix multiplication
    void multiplyGPU(const GF2Matrix& a, const GF2Matrix& b, GF2Matrix& result);
    
    // Performance profiling
    float benchmark(const GF2Matrix& a, const GF2Matrix& b, int iterations = 10);
    
    // Validation
    bool validate(const GF2Matrix& a, const GF2Matrix& b);
    
private:
    MTL::Device* _device;
    MTL::CommandQueue* _commandQueue;
    MTL::ComputePipelineState* _computePipeline;
    
    struct GPUParams {
        uint32_t a_rows;
        uint32_t a_cols;
        uint32_t b_cols;
        uint32_t words_per_row_a;
        uint32_t words_per_row_b;
        uint32_t words_per_row_result;
    };
    
    void setupPipeline();
    MTL::Buffer* createBuffer(const uint64_t* data, size_t size);
    MTL::Buffer* createResultBuffer(size_t size);
};

