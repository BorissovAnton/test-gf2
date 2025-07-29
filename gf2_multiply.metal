/*
GPU kernel for GF(2) matrix multiplication
This kernel performs A × B = C over GF(2) field
*/

#include <metal_stdlib>
using namespace metal;

// Parameters structure
struct GF2Params {
    uint a_rows;
    uint a_cols;
    uint b_cols;
    uint words_per_row_a;
    uint words_per_row_b;
    uint words_per_row_result;
};

// Bit manipulation helpers
inline uint get_bit(device const uint64_t* data, uint row, uint col, uint words_per_row) {
    uint word_idx = col / 64;
    uint bit_idx = col % 64;
    uint64_t word = data[row * words_per_row + word_idx];
    return (word >> bit_idx) & 1ULL;
}

inline void set_bit(device uint64_t* data, uint row, uint col, uint value, uint words_per_row) {
    uint word_idx = col / 64;
    uint bit_idx = col % 64;
    uint64_t mask = 1ULL << bit_idx;
    uint idx = row * words_per_row + word_idx;
    
    if (value) {
        data[idx] |= mask;
    } else {
        data[idx] &= ~mask;
    }
}

// Main multiplication kernel
gf2_multiply_kernel(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b [[buffer(1)]],
    device uint64_t* result [[buffer(2)]],
    constant GF2Params& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    uint row = gid.x;
    uint col = gid.y;
    
    if (row >= params.a_rows || col >= params.b_cols) {
        return;
    }
    
    // Compute GF(2) dot product
    uint sum = 0;
    for (uint k = 0; k < params.a_cols; k++) {
        uint a_bit = get_bit(a, row, k, params.words_per_row_a);
        uint b_bit = get_bit(b, k, col, params.words_per_row_b);
        sum ^= (a_bit & b_bit);
    }
    
    set_bit(result, row, col, sum, params.words_per_row_result);
}

// Optimized kernel for 64-bit aligned data
kernel void gf2_multiply_aligned(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b [[buffer(1)]],
    device uint64_t* result [[buffer(2)]],
    constant GF2Params& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    uint row = gid.x;
    uint word_col = gid.y; // Process 64 columns at a time
    
    if (row >= params.a_rows || word_col >= params.words_per_row_result) {
        return;
    }
    
    uint64_t word_result = 0;
    uint start_col = word_col * 64;
    
    // Process 64 columns simultaneously
    for (uint bit_idx = 0; bit_idx < 64 && start_col + bit_idx < params.b_cols; bit_idx++) {
        uint col = start_col + bit_idx;
        uint sum = 0;
        
        // Compute dot product for this column bit
        for (uint k = 0; k < params.a_cols; k++) {
            uint a_bit = get_bit(a, row, k, params.words_per_row_a);
            uint b_bit = get_bit(b, k, col, params.words_per_row_b);
            sum ^= (a_bit & b_bit);
        }
        
        if (sum) {
            word_result |= (1ULL << bit_idx);
        }
    }
    
    result[row * params.words_per_row_result + word_col] = word_result;
}

// Batch processing kernel for better GPU utilization
kernel void gf2_multiply_batch(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b [[buffer(1)]],
    device uint64_t* result [[buffer(2)]],
    constant GF2Params& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    uint batch_idx = gid.z;
    uint row = gid.x;
    uint word_col = gid.y;
    
    if (batch_idx >= 1) return; // For now, single batch
    
    // Call aligned multiplication
    gf2_multiply_aligned(a, b, result, params, uint2(row, word_col));
}

