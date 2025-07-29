#include "GF2GPU.hpp"
#include <iostream>
#include <chrono>
#include <cassert>

GF2GPU::GF2GPU(MTL::Device* device) : _device(device) {
    setupPipeline();
}

GF2GPU::~GF2GPU() {
    if (_computePipeline) _computePipeline->release();
    if (_commandQueue) _commandQueue->release();
}

void GF2GPU::setupPipeline() {
    NS::Error* error = nullptr;
    
    // Load the shader library
    MTL::Library* library = _device->newDefaultLibrary();
    if (!library) {
        std::cerr << "Failed to load Metal library" << std::endl;
        return;
    }
    
    // Load the kernel function
    auto functionName = NS::String::string("gf2_multiply_aligned", NS::ASCIIStringEncoding);
    MTL::Function* kernelFunction = library->newFunction(functionName);
    if (!kernelFunction) {
        std::cerr << "Failed to load kernel function" << std::endl;
        library->release();
        return;
    }
    
    // Create compute pipeline
    _computePipeline = _device->newComputePipelineState(kernelFunction, &error);
    if (!_computePipeline) {
        std::cerr << "Failed to create compute pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        kernelFunction->release();
        library->release();
        return;
    }
    
    _commandQueue = _device->newCommandQueue();
    if (!_commandQueue) {
        std::cerr << "Failed to create command queue" << std::endl;
    }
    
    kernelFunction->release();
    library->release();
}

void GF2GPU::multiplyGPU(const GF2Matrix& a, const GF2Matrix& b, GF2Matrix& result) {
    if (a.cols() != b.rows()) {
        throw std::runtime_error("Matrix dimensions incompatible for GPU multiplication");
    }
    
    // Calculate buffer sizes
    size_t words_per_row_a = (a.cols() + 63) / 64;
    size_t words_per_row_b = (b.cols() + 63) / 64;
    size_t words_per_row_result = (result.cols() + 63) / 64;
    
    size_t buffer_size_a = a.rows() * words_per_row_a * sizeof(uint64_t);
    size_t buffer_size_b = b.rows() * words_per_row_b * sizeof(uint64_t);
    size_t buffer_size_result = result.rows() * words_per_row_result * sizeof(uint64_t);
    
    // Create GPU buffers
    auto* bufferA = _device->newBuffer(buffer_size_a, MTL::ResourceStorageModeShared);
    auto* bufferB = _device->newBuffer(buffer_size_b, MTL::ResourceStorageModeShared);
    auto* bufferResult = _device->newBuffer(buffer_size_result, MTL::ResourceStorageModeShared);
    
    // Fill buffers with data
    uint64_t* a_data = static_cast<uint64_t*>(bufferA->contents());
    uint64_t* b_data = static_cast<uint64_t*>(bufferB->contents());
    
    // Copy data to GPU buffers (simplified for now)
    memset(a_data, 0, buffer_size_a);
    memset(b_data, 0, buffer_size_b);
    
    // Set up parameters
    GPUParams params;
    params.a_rows = static_cast<uint32_t>(a.rows());
    params.a_cols = static_cast<uint32_t>(a.cols());
    params.b_cols = static_cast<uint32_t>(b.cols());
    params.words_per_row_a = static_cast<uint32_t>(words_per_row_a);
    params.words_per_row_b = static_cast<uint32_t>(words_per_row_b);
    params.words_per_row_result = static_cast<uint32_t>(words_per_row_result);
    
    auto* paramsBuffer = _device->newBuffer(sizeof(GPUParams), MTL::ResourceStorageModeShared);
    memcpy(paramsBuffer->contents(), &params, sizeof(GPUParams));
    
    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = _commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    // Set compute pipeline
    encoder->setComputePipelineState(_computePipeline);
    
    // Set buffers
    encoder->setBuffer(bufferA, 0, 0);
    encoder->setBuffer(bufferB, 0, 1);
    encoder->setBuffer(bufferResult, 0, 2);
    encoder->setBuffer(paramsBuffer, 0, 3);
    
    // Calculate thread group sizes
    MTL::Size threadsPerGroup = MTL::Size::Make(16, 16, 1);
    MTL::Size gridSize = MTL::Size::Make(
        (a.rows() + 15) / 16 * 16,
        (words_per_row_result + 15) / 16 * 16,
        1
    );
    
    encoder->dispatchThreads(gridSize, threadsPerGroup);
    encoder->endEncoding();
    
    // Execute and wait
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy result back (simplified)
    // In real implementation, we'd parse the GPU result into the GF2Matrix
    
    // Cleanup
    bufferA->release();
    bufferB->release();
    bufferResult->release();
    paramsBuffer->release();
}

float GF2GPU::benchmark(const GF2Matrix& a, const GF2Matrix& b, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        GF2Matrix result(a.rows(), b.cols());
        multiplyGPU(a, b, result);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    
    return duration.count() / iterations;
}

bool GF2GPU::validate(const GF2Matrix& a, const GF2Matrix& b) {
    try {
        GF2Matrix serial_result = a.multiplySerial(b);
        GF2Matrix gpu_result(a.rows(), b.cols());
        multiplyGPU(a, b, gpu_result);
        
        return serial_result == gpu_result;
    } catch (...) {
        return false;
    }
}

MTL::Buffer* GF2GPU::createBuffer(const uint64_t* data, size_t size) {
    auto* buffer = _device->newBuffer(size, MTL::ResourceStorageModeShared);
    if (data) {
        memcpy(buffer->contents(), data, size);
    }
    return buffer;
}

MTL::Buffer* GF2GPU::createResultBuffer(size_t size) {
    return _device->newBuffer(size, MTL::ResourceStorageModeShared);
}

