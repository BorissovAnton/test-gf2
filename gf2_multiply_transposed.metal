#include <metal_stdlib>
using namespace metal;

// Use the same parameter struct as the C++ side for consistency.
struct GPUParams {
    uint a_rows;
    uint a_cols; // This is the common dimension, K
    uint b_cols;
    uint words_per_row_a;
    uint words_per_row_b; // For the transposed B matrix
    uint words_per_row_result;
};

// --- FIX STARTS HERE ---

// This is a standard helper function. It is declared without any special
// keyword like 'kernel' or 'function'. The compiler knows it's a helper
// because it's not a kernel, vertex, or fragment function.
void gf2_multiply_transposed_logic(
    device const uint64_t* a,
    device const uint64_t* b_transposed,
    device uint64_t* result,
    constant GPUParams& params,
    uint2 gid) // The grid position is passed as a normal argument.
{
    uint c_row = gid.x;
    uint c_word_col = gid.y;

    if (c_row >= params.a_rows || c_word_col >= params.words_per_row_result) {
        return;
    }

    // Pointer to the start of the relevant row in A.
    device const uint64_t* a_row_ptr = a + c_row * params.words_per_row_a;
    
    uint64_t result_word = 0;
    uint common_dim_words = params.a_cols / 64;

    // This outer loop calculates each of the 64 bits for our target result word.
    for (uint bit_idx = 0; bit_idx < 64; ++bit_idx) {
        // The original column in B we are targeting.
        uint b_col_original = c_word_col * 64 + bit_idx;
        if (b_col_original >= params.b_cols) continue;

        // In the transposed matrix, this corresponds to a full row.
        device const uint64_t* b_t_row_ptr = b_transposed + b_col_original * params.words_per_row_b;

        // --- Perform the dot product for a single result bit ---
        uint64_t dot_product_acc = 0;
        for (uint k = 0; k < common_dim_words; ++k) {
            dot_product_acc ^= (a_row_ptr[k] & b_t_row_ptr[k]);
        }
        
        // Horizontal reduction (parity check) to get the final bit.
        if (__builtin_popcount(dot_product_acc) % 2) {
            result_word |= (1ULL << bit_idx);
        }
    }

    result[c_row * params.words_per_row_result + c_word_col] = result_word;
}


// This is the KERNEL, the single entry point for the CPU to call.
// It receives the buffers and grid position from the host.
kernel void gf2_multiply_transposed_batch(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b_transposed [[buffer(1)]],
    device uint64_t* result [[buffer(2)]],
    constant GPUParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    // For now, single batch. This structure allows for future expansion.
    if (gid.z >= 1) return;
    
    // It now correctly calls the helper function, passing along the arguments.
    gf2_multiply_transposed_logic(a, b_transposed, result, params, gid.xy);
}

// --- FIX ENDS HERE ---

