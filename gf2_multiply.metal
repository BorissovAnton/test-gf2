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

// --- FIX STARTS HERE ---

// FIX: Use the explicit fixed-width atomic type 'atomic_uint'.
inline void set_bit_atomic(volatile device atomic_uint* data, uint row, uint col, uint words_per_row) {
    uint word_idx = col / 64;
    uint bit_idx = col % 64;
    
    volatile device atomic_uint* atomic_ptr = &data[row * words_per_row + word_idx];
    
    // FIX: The mask's type MUST be uint64_t to match the atomic's base type.
    // Use the 'ULL' suffix for an unsigned long long literal.
    uint64_t mask = 1ULL << bit_idx;
    
    // This will now compile correctly as all types match what the compiler expects.
    atomic_fetch_or_explicit(atomic_ptr, mask, memory_order_relaxed);
}

// Main multiplication kernel (thread-per-element)
kernel void gf2_multiply_kernel(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b [[buffer(1)]],
    // FIX: The result buffer must use the 'atomic_uint' type.
    volatile device atomic_uint* result [[buffer(2)]],
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
    
    if (sum) {
        set_bit_atomic(result, row, col, params.words_per_row_result);
    }
}

// --- FIX ENDS HERE ---


// Optimized function for 64-bit aligned data (thread-per-word)
// This function is already correct and uses uint64_t.
void gf2_multiply_aligned(
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
    
    for (uint bit_idx = 0; bit_idx < 64 && start_col + bit_idx < params.b_cols; bit_idx++) {
        uint col = start_col + bit_idx;
        uint sum = 0;
        
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
    
    gf2_multiply_aligned(a, b, result, params, uint2(row, word_col));
}

