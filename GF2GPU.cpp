#include "GF2GPU.hpp"
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>

// Constructor with correct initializer list order
GF2GPU::GF2GPU(MTL::Device *device)
    : _device(device), _commandQueue(nullptr), _computePipeline(nullptr),
      _computePipelineTransposed(nullptr), _computePipelineTiled(nullptr),
      _computePipelineVectorized(nullptr),
      // --- NEW: Initialize M4R pipeline pointers ---
      _computePipelineM4R_MakeTable(nullptr),
      _computePipelineM4R_Multiply(nullptr)
{
  setupPipeline();
}

GF2GPU::~GF2GPU() {
  if (_computePipeline)
    _computePipeline->release();
  if (_computePipelineTransposed)
    _computePipelineTransposed->release();
  if (_computePipelineTiled)
    _computePipelineTiled->release();
  if (_computePipelineVectorized)
    _computePipelineVectorized->release();
  // --- NEW: Release M4R pipeline states ---
  if (_computePipelineM4R_MakeTable)
    _computePipelineM4R_MakeTable->release();
  if (_computePipelineM4R_Multiply)
    _computePipelineM4R_Multiply->release();
  if (_commandQueue)
    _commandQueue->release();
}

void GF2GPU::setupPipeline() {
  NS::Error *error = nullptr;

  MTL::Library *library = _device->newDefaultLibrary();
  if (!library) {
    std::cerr << "Failed to load Metal library" << std::endl;
    return;
  }

  // --- Setup for original kernel ---
  auto functionName =
      NS::String::string("gf2_multiply_batch", NS::ASCIIStringEncoding);
  MTL::Function *kernelFunction = library->newFunction(functionName);
  if (kernelFunction) {
    _computePipeline = _device->newComputePipelineState(kernelFunction, &error);
    if (!_computePipeline)
      std::cerr << "Failed to create pipeline for original kernel: "
                << error->localizedDescription()->utf8String() << std::endl;
    kernelFunction->release();
  } else {
    std::cerr << "Failed to load kernel function: gf2_multiply_batch"
              << std::endl;
  }

  // --- Setup for transposed kernel ---
  auto functionNameTransposed = NS::String::string(
      "gf2_multiply_transposed_batch", NS::ASCIIStringEncoding);
  MTL::Function *kernelFunctionTransposed =
      library->newFunction(functionNameTransposed);
  if (kernelFunctionTransposed) {
    _computePipelineTransposed =
        _device->newComputePipelineState(kernelFunctionTransposed, &error);
    if (!_computePipelineTransposed)
      std::cerr << "Failed to create pipeline for transposed kernel: "
                << error->localizedDescription()->utf8String() << std::endl;
    kernelFunctionTransposed->release();
  } else {
    std::cerr << "Failed to load kernel function: gf2_multiply_transposed_batch"
              << std::endl;
  }

  // --- Setup for tiled kernel ---
  auto functionNameTiled =
      NS::String::string("gf2_multiply_tiled_kernel", NS::ASCIIStringEncoding);
  MTL::Function *kernelFunctionTiled = library->newFunction(functionNameTiled);
  if (kernelFunctionTiled) {
    _computePipelineTiled =
        _device->newComputePipelineState(kernelFunctionTiled, &error);
    if (!_computePipelineTiled)
      std::cerr << "Failed to create pipeline for tiled kernel: "
                << error->localizedDescription()->utf8String() << std::endl;
    kernelFunctionTiled->release();
  } else {
    std::cerr << "Failed to load kernel function: gf2_multiply_tiled_kernel"
              << std::endl;
  }

  // --- Setup for vectorized kernel ---
  auto functionNameVectorized = NS::String::string(
      "gf2_multiply_vectorized_batch", NS::ASCIIStringEncoding);
  MTL::Function *kernelFunctionVectorized =
      library->newFunction(functionNameVectorized);
  if (kernelFunctionVectorized) {
    _computePipelineVectorized =
        _device->newComputePipelineState(kernelFunctionVectorized, &error);
    if (!_computePipelineVectorized)
      std::cerr << "Failed to create pipeline for vectorized kernel: "
                << error->localizedDescription()->utf8String() << std::endl;
    kernelFunctionVectorized->release();
  } else {
    std::cerr << "Failed to load kernel function: gf2_multiply_vectorized_batch"
              << std::endl;
  }

  // --- NEW: Setup for M4R kernels ---
  auto functionNameM4R_Table =
      NS::String::string("m4r_make_tables_kernel", NS::ASCIIStringEncoding);
  MTL::Function *kernelFunctionM4R_Table = library->newFunction(functionNameM4R_Table);
  if (kernelFunctionM4R_Table) {
    _computePipelineM4R_MakeTable =
        _device->newComputePipelineState(kernelFunctionM4R_Table, &error);
    if (!_computePipelineM4R_MakeTable)
      std::cerr << "Failed to create pipeline for M4R table kernel: "
                << error->localizedDescription()->utf8String() << std::endl;
    kernelFunctionM4R_Table->release();
  } else {
    std::cerr << "Failed to load kernel function: m4r_make_tables_kernel"
              << std::endl;
  }

  auto functionNameM4R_Mul =
      NS::String::string("m4r_multiply_kernel", NS::ASCIIStringEncoding);
  MTL::Function *kernelFunctionM4R_Mul = library->newFunction(functionNameM4R_Mul);
  if (kernelFunctionM4R_Mul) {
    _computePipelineM4R_Multiply =
        _device->newComputePipelineState(kernelFunctionM4R_Mul, &error);
    if (!_computePipelineM4R_Multiply)
      std::cerr << "Failed to create pipeline for M4R multiply kernel: "
                << error->localizedDescription()->utf8String() << std::endl;
    kernelFunctionM4R_Mul->release();
  } else {
    std::cerr << "Failed to load kernel function: m4r_multiply_kernel"
              << std::endl;
  }

  _commandQueue = _device->newCommandQueue();
  if (!_commandQueue) {
    std::cerr << "Failed to create command queue" << std::endl;
  }

  library->release();
}

// --- NEW: Implementation for the M4R multiplication method ---
void GF2GPU::multiplyGPUM4R(const GF2Matrix &a, const GF2Matrix &b,
                            GF2Matrix &result) {
  if (a.cols() != b.rows() || !_computePipelineM4R_MakeTable || !_computePipelineM4R_Multiply) {
    throw std::runtime_error(
        "Matrix dimensions incompatible or M4R pipelines not ready.");
  }

  // --- M4R Constants ---
  const size_t K_M4R = 8;
  const size_t TABLE_ROWS = 1 << K_M4R; // 256
  const size_t CHUNKS_PER_WORD = 64 / K_M4R; // 8

  // --- Buffer and Table Size Calculations ---
  size_t num_tables = a.words_per_row() * CHUNKS_PER_WORD;
  size_t table_row_size_words = b.words_per_row();
  size_t single_table_size_bytes = TABLE_ROWS * table_row_size_words * sizeof(uint64_t);
  size_t total_table_size_bytes = num_tables * single_table_size_bytes;

  size_t buffer_size_a = a.rows() * a.words_per_row() * sizeof(uint64_t);
  size_t buffer_size_b = b.rows() * b.words_per_row() * sizeof(uint64_t);
  size_t buffer_size_result = result.rows() * result.words_per_row() * sizeof(uint64_t);

  // --- Create Metal Buffers ---
  auto *bufferA = _device->newBuffer(a.get_raw_data(), buffer_size_a, MTL::ResourceStorageModeShared);
  auto *bufferB = _device->newBuffer(b.get_raw_data(), buffer_size_b, MTL::ResourceStorageModeShared);
  auto *bufferResult = _device->newBuffer(buffer_size_result, MTL::ResourceStorageModeShared);
  auto *bufferLookupTables = _device->newBuffer(total_table_size_bytes, MTL::ResourceStorageModeShared);

  GPUParams params;
  params.a_rows = static_cast<uint32_t>(a.rows());
  params.a_cols = static_cast<uint32_t>(a.cols());
  params.b_cols = static_cast<uint32_t>(b.cols());
  params.words_per_row_a = static_cast<uint32_t>(a.words_per_row());
  params.words_per_row_b = static_cast<uint32_t>(b.words_per_row());
  params.words_per_row_result = static_cast<uint32_t>(result.words_per_row());

  auto *paramsBuffer = _device->newBuffer(&params, sizeof(GPUParams), MTL::ResourceStorageModeShared);

  // --- Command Dispatch ---
  MTL::CommandBuffer *commandBuffer = _commandQueue->commandBuffer();

  // --- Pass 1: Generate Lookup Tables ---
  MTL::ComputeCommandEncoder *tableEncoder = commandBuffer->computeCommandEncoder();
  tableEncoder->setComputePipelineState(_computePipelineM4R_MakeTable);
  tableEncoder->setBuffer(bufferB, 0, 0);
  tableEncoder->setBuffer(bufferLookupTables, 0, 1);
  tableEncoder->setBuffer(paramsBuffer, 0, 2);

  MTL::Size tableGridSize = MTL::Size::Make(b.words_per_row(), num_tables, 1);
  MTL::Size tableGroupSize = MTL::Size::Make(16, 16, 1); // A common, safe threadgroup size
  tableEncoder->dispatchThreads(tableGridSize, tableGroupSize);
  tableEncoder->endEncoding();

  // --- Pass 2: Perform Multiplication ---
  MTL::ComputeCommandEncoder *mulEncoder = commandBuffer->computeCommandEncoder();
  mulEncoder->setComputePipelineState(_computePipelineM4R_Multiply);
  mulEncoder->setBuffer(bufferA, 0, 0);
  mulEncoder->setBuffer(bufferResult, 0, 1);
  mulEncoder->setBuffer(bufferLookupTables, 0, 2);
  mulEncoder->setBuffer(paramsBuffer, 0, 3);

  MTL::Size mulGridSize = MTL::Size::Make(a.rows(), result.words_per_row(), 1);
  MTL::Size mulGroupSize = MTL::Size::Make(16, 16, 1);
  mulEncoder->dispatchThreads(mulGridSize, mulGroupSize);
  mulEncoder->endEncoding();

  // --- Finalize ---
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  memcpy(const_cast<uint64_t *>(result.get_raw_data()), bufferResult->contents(), buffer_size_result);

  // --- Release Buffers ---
  bufferA->release();
  bufferB->release();
  bufferResult->release();
  bufferLookupTables->release();
  paramsBuffer->release();
}


// --- The other multiplication methods remain unchanged ---

void GF2GPU::multiplyGPUVectorized(const GF2Matrix &a, const GF2Matrix &b,
                                   GF2Matrix &result) {
  if (a.cols() != b.rows() || !_computePipelineVectorized) {
    throw std::runtime_error(
        "Matrix dimensions incompatible or vectorized pipeline not ready.");
  }

  // This method also relies on the transposed B matrix for coalesced memory
  // access.
  GF2Matrix b_t = b.transpose();

  size_t buffer_size_a = a.rows() * a.words_per_row() * sizeof(uint64_t);
  size_t buffer_size_b_t = b_t.rows() * b_t.words_per_row() * sizeof(uint64_t);
  size_t buffer_size_result =
      result.rows() * result.words_per_row() * sizeof(uint64_t);

  auto *bufferA = _device->newBuffer(a.get_raw_data(), buffer_size_a,
                                     MTL::ResourceStorageModeShared);
  auto *bufferB_T = _device->newBuffer(b_t.get_raw_data(), buffer_size_b_t,
                                       MTL::ResourceStorageModeShared);
  auto *bufferResult =
      _device->newBuffer(buffer_size_result, MTL::ResourceStorageModeShared);

  GPUParams params;
  params.a_rows = static_cast<uint32_t>(a.rows());
  params.a_cols = static_cast<uint32_t>(a.cols());
  params.b_cols = static_cast<uint32_t>(b.cols());
  params.words_per_row_a = static_cast<uint32_t>(a.words_per_row());
  params.words_per_row_b = static_cast<uint32_t>(b_t.words_per_row());
  params.words_per_row_result = static_cast<uint32_t>(result.words_per_row());

  auto *paramsBuffer = _device->newBuffer(&params, sizeof(GPUParams),
                                          MTL::ResourceStorageModeShared);

  MTL::CommandBuffer *commandBuffer = _commandQueue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  // Set the VECTORIZED compute pipeline
  encoder->setComputePipelineState(_computePipelineVectorized);

  encoder->setBuffer(bufferA, 0, 0);
  encoder->setBuffer(bufferB_T, 0, 1);
  encoder->setBuffer(bufferResult, 0, 2);
  encoder->setBuffer(paramsBuffer, 0, 3);

  // The dispatch grid is the same as the transposed version.
  // Each thread computes one uint64_t word of the result.
  MTL::Size threadsPerGroup = MTL::Size::Make(16, 16, 1);
  MTL::Size gridSize = MTL::Size::Make(a.rows(), result.words_per_row(), 1);

  encoder->dispatchThreads(gridSize, threadsPerGroup);
  encoder->endEncoding();

  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  memcpy(const_cast<uint64_t *>(result.get_raw_data()),
         bufferResult->contents(), buffer_size_result);

  bufferA->release();
  bufferB_T->release();
  bufferResult->release();
  paramsBuffer->release();
}

void GF2GPU::multiplyGPU_transposed(const GF2Matrix &a, const GF2Matrix &b,
                                    GF2Matrix &result) {
  if (a.cols() != b.rows() || !_computePipelineTransposed) {
    throw std::runtime_error(
        "Matrix dimensions incompatible or transposed pipeline not ready.");
  }
  GF2Matrix b_t = b.transpose();
  size_t buffer_size_a = a.rows() * a.words_per_row() * sizeof(uint64_t);
  size_t buffer_size_b_t = b_t.rows() * b_t.words_per_row() * sizeof(uint64_t);
  size_t buffer_size_result =
      result.rows() * result.words_per_row() * sizeof(uint64_t);
  auto *bufferA = _device->newBuffer(a.get_raw_data(), buffer_size_a,
                                     MTL::ResourceStorageModeShared);
  auto *bufferB_T = _device->newBuffer(b_t.get_raw_data(), buffer_size_b_t,
                                       MTL::ResourceStorageModeShared);
  auto *bufferResult =
      _device->newBuffer(buffer_size_result, MTL::ResourceStorageModeShared);
  GPUParams params;
  params.a_rows = static_cast<uint32_t>(a.rows());
  params.a_cols = static_cast<uint32_t>(a.cols());
  params.b_cols = static_cast<uint32_t>(b.cols());
  params.words_per_row_a = static_cast<uint32_t>(a.words_per_row());
  params.words_per_row_b = static_cast<uint32_t>(b_t.words_per_row());
  params.words_per_row_result = static_cast<uint32_t>(result.words_per_row());
  auto *paramsBuffer = _device->newBuffer(&params, sizeof(GPUParams),
                                          MTL::ResourceStorageModeShared);
  MTL::CommandBuffer *commandBuffer = _commandQueue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(_computePipelineTransposed);
  encoder->setBuffer(bufferA, 0, 0);
  encoder->setBuffer(bufferB_T, 0, 1);
  encoder->setBuffer(bufferResult, 0, 2);
  encoder->setBuffer(paramsBuffer, 0, 3);
  MTL::Size threadsPerGroup = MTL::Size::Make(16, 16, 1);
  MTL::Size gridSize = MTL::Size::Make(a.rows(), result.words_per_row(), 1);
  encoder->dispatchThreads(gridSize, threadsPerGroup);
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  memcpy(const_cast<uint64_t *>(result.get_raw_data()),
         bufferResult->contents(), buffer_size_result);
  bufferA->release();
  bufferB_T->release();
  bufferResult->release();
  paramsBuffer->release();
}

void GF2GPU::multiplyGPUTiled(const GF2Matrix &a, const GF2Matrix &b,
                              GF2Matrix &result) {
  if (a.cols() != b.rows()) {
    throw std::runtime_error(
        "Matrix dimensions incompatible for GPU multiplication");
  }
  if (!_computePipelineTiled) {
    throw std::runtime_error("Tiled GPU pipeline not initialized.");
  }
  size_t words_per_row_a = a.words_per_row();
  size_t words_per_row_b = b.words_per_row();
  size_t words_per_row_result = result.words_per_row();
  size_t buffer_size_a = a.rows() * words_per_row_a * sizeof(uint64_t);
  size_t buffer_size_b = b.rows() * words_per_row_b * sizeof(uint64_t);
  size_t buffer_size_result =
      result.rows() * words_per_row_result * sizeof(uint64_t);
  auto *bufferA = _device->newBuffer(a.get_raw_data(), buffer_size_a,
                                     MTL::ResourceStorageModeShared);
  auto *bufferB = _device->newBuffer(b.get_raw_data(), buffer_size_b,
                                     MTL::ResourceStorageModeShared);
  auto *bufferResult =
      _device->newBuffer(buffer_size_result, MTL::ResourceStorageModeShared);
  GPUParams params;
  params.a_rows = static_cast<uint32_t>(a.rows());
  params.a_cols = static_cast<uint32_t>(a.cols());
  params.b_cols = static_cast<uint32_t>(b.cols());
  params.words_per_row_a = static_cast<uint32_t>(words_per_row_a);
  params.words_per_row_b = static_cast<uint32_t>(words_per_row_b);
  params.words_per_row_result = static_cast<uint32_t>(words_per_row_result);
  auto *paramsBuffer = _device->newBuffer(&params, sizeof(GPUParams),
                                          MTL::ResourceStorageModeShared);
  MTL::CommandBuffer *commandBuffer = _commandQueue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(_computePipelineTiled);
  encoder->setBuffer(bufferA, 0, 0);
  encoder->setBuffer(bufferB, 0, 1);
  encoder->setBuffer(bufferResult, 0, 2);
  encoder->setBuffer(paramsBuffer, 0, 3);
  const int TILE_WIDTH = 32;
  MTL::Size threadsPerGroup = MTL::Size::Make(TILE_WIDTH, TILE_WIDTH, 1);
  MTL::Size gridSize = MTL::Size::Make(a.rows(), b.cols(), 1);
  encoder->dispatchThreads(gridSize, threadsPerGroup);
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  memcpy(const_cast<uint64_t *>(result.get_raw_data()),
         bufferResult->contents(), buffer_size_result);
  bufferA->release();
  bufferB->release();
  bufferResult->release();
  paramsBuffer->release();
}

void GF2GPU::multiplyGPU(const GF2Matrix &a, const GF2Matrix &b,
                         GF2Matrix &result) {
  if (a.cols() != b.rows()) {
    throw std::runtime_error(
        "Matrix dimensions incompatible for GPU multiplication");
  }
  size_t words_per_row_a = a.words_per_row();
  size_t words_per_row_b = b.words_per_row();
  size_t words_per_row_result = result.words_per_row();
  size_t buffer_size_a = a.rows() * words_per_row_a * sizeof(uint64_t);
  size_t buffer_size_b = b.rows() * words_per_row_b * sizeof(uint64_t);
  size_t buffer_size_result =
      result.rows() * words_per_row_result * sizeof(uint64_t);
  auto *bufferA = _device->newBuffer(a.get_raw_data(), buffer_size_a,
                                     MTL::ResourceStorageModeShared);
  auto *bufferB = _device->newBuffer(b.get_raw_data(), buffer_size_b,
                                     MTL::ResourceStorageModeShared);
  auto *bufferResult =
      _device->newBuffer(buffer_size_result, MTL::ResourceStorageModeShared);
  GPUParams params;
  params.a_rows = static_cast<uint32_t>(a.rows());
  params.a_cols = static_cast<uint32_t>(a.cols());
  params.b_cols = static_cast<uint32_t>(b.cols());
  params.words_per_row_a = static_cast<uint32_t>(words_per_row_a);
  params.words_per_row_b = static_cast<uint32_t>(words_per_row_b);
  params.words_per_row_result = static_cast<uint32_t>(words_per_row_result);
  auto *paramsBuffer = _device->newBuffer(&params, sizeof(GPUParams),
                                          MTL::ResourceStorageModeShared);
  MTL::CommandBuffer *commandBuffer = _commandQueue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(_computePipeline);
  encoder->setBuffer(bufferA, 0, 0);
  encoder->setBuffer(bufferB, 0, 1);
  encoder->setBuffer(bufferResult, 0, 2);
  encoder->setBuffer(paramsBuffer, 0, 3);
  MTL::Size threadsPerGroup = MTL::Size::Make(16, 16, 1);
  MTL::Size gridSize = MTL::Size::Make(a.rows(), words_per_row_result, 1);
  encoder->dispatchThreads(gridSize, threadsPerGroup);
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  memcpy(const_cast<uint64_t *>(result.get_raw_data()),
         bufferResult->contents(), buffer_size_result);
  bufferA->release();
  bufferB->release();
  bufferResult->release();
  paramsBuffer->release();
}

// Benchmark and other helpers remain unchanged
float GF2GPU::benchmark(const GF2Matrix &a, const GF2Matrix &b,
                        int iterations) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    GF2Matrix result(a.rows(), b.cols());
    multiplyGPU(a, b, result);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration = end - start;
  return duration.count() / iterations;
}

bool GF2GPU::validate(const GF2Matrix &a, const GF2Matrix &b) {
  try {
    GF2Matrix serial_result = a.multiplySerial(b);
    GF2Matrix gpu_result(a.rows(), b.cols());
    multiplyGPU(a, b, gpu_result);
    return serial_result == gpu_result;
  } catch (...) {
    return false;
  }
}

MTL::Buffer *GF2GPU::createBuffer(const uint64_t *data, size_t size) {
  auto *buffer = _device->newBuffer(size, MTL::ResourceStorageModeShared);
  if (data) {
    memcpy(buffer->contents(), data, size);
  }
  return buffer;
}

MTL::Buffer *GF2GPU::createResultBuffer(size_t size) {
  return _device->newBuffer(size, MTL::ResourceStorageModeShared);
}

