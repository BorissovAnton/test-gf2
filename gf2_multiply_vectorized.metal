// --- File: gf2_multiply_vectorized.metal ---

#include <metal_stdlib>
using namespace metal;

// Use the same parameter struct as the C++ side for consistency.
// This is identical to the struct in gf2_multiply_transposed.metal.
struct GPUParams {
    uint a_rows;
    uint a_cols; // This is the common dimension, K
    uint b_cols;
    uint words_per_row_a;
    uint words_per_row_b; // For the transposed B matrix
    uint words_per_row_result;
};

// This is the logic function that performs the vectorized multiplication.
// It's designed to be called by the main kernel entry point.
void gf2_multiply_vectorized_logic(
    device const uint64_t* a,
    device const uint64_t* b_transposed,
    device uint64_t* result,
    constant GPUParams& params,
    uint2 gid)
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
        uint b_col_original = c_word_col * 64 + bit_idx;
        if (b_col_original >= params.b_cols) continue;

        // In the transposed matrix, this corresponds to a full row.
        device const uint64_t* b_t_row_ptr = b_transposed + b_col_original * params.words_per_row_b;

        // --- Vectorized Dot Product for a single result bit ---
        
        // 1. Vectorized part: Use a 256-bit accumulator (4 x 64-bit).
        ulong4 acc_vec = (ulong4)(0);
        uint common_dim_vecs = common_dim_words / 4;

        for (uint k_vec = 0; k_vec < common_dim_vecs; ++k_vec) {
            uint k_offset = k_vec * 4;
            // Load 256 bits (4 words) from A and B_transposed at once.
            ulong4 a_vec = *(device ulong4*)(a_row_ptr + k_offset);
            ulong4 b_vec = *(device ulong4*)(b_t_row_ptr + k_offset);
            
            // Perform 256 ANDs and 256 XORs in a single instruction.
            acc_vec ^= (a_vec & b_vec);
        }
        
        // 2. Horizontal Reduction: XOR the 4 lanes of the vector accumulator together.
        uint64_t dot_product_acc = acc_vec.x ^ acc_vec.y ^ acc_vec.z ^ acc_vec.w;

        // 3. Scalar part: Handle any remaining words if common_dim_words is not a multiple of 4.
        for (uint k = common_dim_vecs * 4; k < common_dim_words; ++k) {
            dot_product_acc ^= (a_row_ptr[k] & b_t_row_ptr[k]);
        }
        
        // 4. Final parity check to get the single result bit.
        if (__builtin_popcount(dot_product_acc) % 2) {
            result_word |= (1ULL << bit_idx);
        }
    }

    result[c_row * params.words_per_row_result + c_word_col] = result_word;
}


// This is the KERNEL, the single entry point for the CPU to call.
kernel void gf2_multiply_vectorized_batch(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b_transposed [[buffer(1)]],
    device uint64_t* result [[buffer(2)]],
    constant GPUParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    // For now, single batch. This structure allows for future expansion.
    if (gid.z >= 1) return;
    
    // Call the vectorized logic function.
    gf2_multiply_vectorized_logic(a, b_transposed, result, params, gid.xy);
}

