// Only compile this file for aarch64 architecture
#if defined(__aarch64__)

#include "GF2Matrix.hpp"
#include <arm_neon.h>
#include <vector>

// This function is declared in GF2Matrix.cpp and called by multiplySIMD
void multiply_simd_neon(const GF2Matrix& a, const GF2Matrix& b, GF2Matrix& result) {
    // Transposing the 'b' matrix is crucial for performance. It allows us to
    // process both matrices row-by-row, which is ideal for SIMD memory access.
    GF2Matrix b_transposed = b.transpose();

    size_t a_rows = a.rows();
    size_t b_cols = b.cols(); // This is equal to b_transposed.rows()
    size_t common_dim_words = a.words_per_row();

    const uint64_t* a_data = a.get_raw_data();
    const uint64_t* b_t_data = b_transposed.get_raw_data();

    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t j = 0; j < b_cols; ++j) {
            const uint64_t* a_row_ptr = a_data + i * common_dim_words;
            const uint64_t* b_t_row_ptr = b_t_data + j * common_dim_words;

            // Use 128-bit NEON vectors, which can hold two 64-bit integers.
            uint64x2_t acc = vdupq_n_u64(0);

            // Process 2 words (128 bits) at a time.
            size_t k = 0;
            for (; k + 1 < common_dim_words; k += 2) {
                uint64x2_t a_vec = vld1q_u64(a_row_ptr + k);
                uint64x2_t b_vec = vld1q_u64(b_t_row_ptr + k);
                
                // Bitwise AND followed by a vector XOR.
                uint64x2_t and_res = vandq_u64(a_vec, b_vec);
                acc = veorq_u64(acc, and_res);
            }

            // Horizontally XOR the two 64-bit lanes of the accumulator vector.
            uint64_t temp_res = vgetq_lane_u64(acc, 0) ^ vgetq_lane_u64(acc, 1);

            // Handle the last word if the number of words is odd.
            if (k < common_dim_words) {
                temp_res ^= (a_row_ptr[k] & b_t_row_ptr[k]);
            }

            // The final result bit is the parity of the total number of set bits.
            // __builtin_popcountll is a highly optimized compiler intrinsic for this.
            result.set(i, j, __builtin_popcountll(temp_res) % 2);
        }
    }
}

#endif // defined(__aarch64__)

