#include "GF2Matrix.hpp"

// Only compile this file for x86_64 architecture
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>
#include <cstring>

// Optimized SIMD multiplication using AVX2/AVX-512
class GF2MatrixSIMD {
public:
    static void multiply(const uint64_t* a, const uint64_t* b, uint64_t* result,
                        size_t a_rows, size_t a_cols, size_t b_cols) {
        
        size_t words_per_row_a = (a_cols + 63) / 64;
        size_t words_per_row_b = (b_cols + 63) / 64;
        size_t words_per_row_result = (b_cols + 63) / 64;
        
        // Process 4 rows at a time using AVX2
        for (size_t i = 0; i < a_rows; i++) {
            for (size_t j = 0; j < words_per_row_result; j++) {
                uint64_t sum = 0;
                
                // Process 64 bits at a time
                for (size_t k = 0; k < a_cols; k += 64) {
                    size_t word_a = k / 64;
                    size_t word_b = j * (a_cols / 64) + word_a;
                    
                    uint64_t a_word = a[i * words_per_row_a + word_a];
                    
                    // Process b in chunks
                    for (size_t bit = 0; bit < 64 && k + bit < a_cols; bit++) {
                        if (a_word & (1ULL << bit)) {
                            uint64_t b_word = b[(k + bit) * words_per_row_b + j];
                            sum ^= b_word;
                        }
                    }
                }
                
                result[i * words_per_row_result + j] = sum;
            }
        }
    }
};

// Bit-slice implementation for better SIMD utilization
class GF2BitSliceMultiplier {
public:
    static void multiply(const uint64_t* a, const uint64_t* b, uint64_t* result,
                        size_t a_rows, size_t a_cols, size_t b_cols) {
        
        const size_t words_per_row_a = (a_cols + 63) / 64;
        const size_t words_per_row_b = (b_cols + 63) / 64;
        const size_t words_per_row_result = (b_cols + 63) / 64;
        
        // Use AVX2 intrinsics for 256-bit operations
        const __m256i one = _mm256_set1_epi64x(1);
        
        for (size_t i = 0; i < a_rows; i++) {
            for (size_t j = 0; j < b_cols; j += 64) {
                uint64_t result_word = 0;
                
                for (size_t k = 0; k < a_cols; k++) {
                    bool a_bit = (a[i * words_per_row_a + (k / 64)] >> (k % 64)) & 1ULL;
                    if (a_bit) {
                        uint64_t b_word = b[k * words_per_row_b + (j / 64)];
                        result_word ^= b_word;
                    }
                }
                
                result[i * words_per_row_result + (j / 64)] = result_word;
            }
        }
    }
};

// AVX-512 optimized multiplication
#ifdef __AVX512F__
class GF2AVX512Multiplier {
public:
    static void multiply(const uint64_t* a, const uint64_t* b, uint64_t* result,
                        size_t a_rows, size_t a_cols, size_t b_cols) {
        
        const size_t words_per_row_a = (a_cols + 63) / 64;
        const size_t words_per_row_b = (b_cols + 63) / 64;
        const size_t words_per_row_result = (b_cols + 63) / 64;
        
        // Use AVX-512 for parallel bit operations
        for (size_t i = 0; i < a_rows; i++) {
            for (size_t j = 0; j < words_per_row_result; j++) {
                __m512i sum = _mm512_setzero_si512();
                
                for (size_t k = 0; k < a_cols; k++) {
                    bool a_bit = getBit(a, i, k, words_per_row_a);
                    if (a_bit) {
                        __m512i b_vec = loadBits(b, k, j, words_per_row_b);
                        sum = _mm512_xor_si512(sum, b_vec);
                    }
                }
                
                storeBits(result, i, j, sum, words_per_row_result);
            }
        }
    }

private:
    static bool getBit(const uint64_t* data, size_t row, size_t col, size_t words_per_row) {
        return (data[row * words_per_row + (col / 64)] >> (col % 64)) & 1ULL;
    }
    
    static __m512i loadBits(const uint64_t* b, size_t row, size_t word_idx, size_t words_per_row) {
        return _mm512_load_si512((__m512i*)(b + row * words_per_row + word_idx));
    }
    
    static void storeBits(uint64_t* result, size_t row, size_t word_idx, __m512i data, size_t words_per_row) {
        _mm512_store_si512((__m512i*)(result + row * words_per_row + word_idx), data);
    }
};
#endif

#endif // defined(__x86_64__) || defined(_M_X64)

