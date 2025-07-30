// Only compile this file for x86_64 architecture
#if defined(__x86_64__) || defined(_M_X64)

#include "GF2Matrix.hpp"
#include <immintrin.h>
#include <vector>

// This function is declared in GF2Matrix.cpp and called by multiplySIMD
void multiply_simd_x86(const GF2Matrix& a, const GF2Matrix& b, GF2Matrix& result) {
    // Use the same transpose optimization for the x86 implementation
    GF2Matrix b_transposed = b.transpose();

    size_t a_rows = a.rows();
    size_t b_cols = b.cols();
    size_t common_dim_words = a.words_per_row();

    const uint64_t* a_data = a.get_raw_data();
    const uint64_t* b_t_data = b_transposed.get_raw_data();

    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t j = 0; j < b_cols; ++j) {
            const uint64_t* a_row_ptr = a_data + i * common_dim_words;
            const uint64_t* b_t_row_ptr = b_t_data + j * common_dim_words;

            // Use 256-bit AVX2 vectors (four 64-bit integers)
            __m256i acc = _mm256_setzero_si256();

            size_t k = 0;
            for (; k + 3 < common_dim_words; k += 4) {
                __m256i a_vec = _mm256_loadu_si256((__m256i*)(a_row_ptr + k));
                __m256i b_vec = _mm256_loadu_si256((__m256i*)(b_t_row_ptr + k));
                __m256i and_res = _mm256_and_si256(a_vec, b_vec);
                acc = _mm256_xor_si256(acc, and_res);
            }

            // Horizontal XOR reduction of the accumulator
            __m128i xmm_acc_low = _mm256_extracti128_si256(acc, 0);
            __m128i xmm_acc_high = _mm256_extracti128_si256(acc, 1);
            __m128i xmm_acc = _mm_xor_si128(xmm_acc_low, xmm_acc_high);
            
            uint64_t temp_res = _mm_extract_epi64(xmm_acc, 0) ^ _mm_extract_epi64(xmm_acc, 1);

            // Handle remaining words
            for (; k < common_dim_words; ++k) {
                temp_res ^= (a_row_ptr[k] & b_t_row_ptr[k]);
            }

            // Final parity check
            result.set(i, j, _mm_popcnt_u64(temp_res) % 2);
        }
    }
}

#endif // defined(__x86_64__) || defined(_M_X64)

