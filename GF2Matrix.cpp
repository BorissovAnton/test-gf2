#include "GF2Matrix.hpp"
#include <iostream>
#include <cstring>
// #include <immintrin.h> 

GF2Matrix::GF2Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
    // Each row is padded to be a multiple of 64 bits for efficient SIMD access.
    m_words_per_row = (cols + 63) / 64;
    m_data.resize(rows * m_words_per_row, 0);
}

bool GF2Matrix::get(size_t row, size_t col) const {
    if (row >= m_rows || col >= m_cols) return false;
    
    size_t word_index = row * m_words_per_row + (col / 64);
    size_t bit_index = col % 64;
    
    return (m_data[word_index] >> bit_index) & 1ULL;
}

void GF2Matrix::set(size_t row, size_t col, bool value) {
    if (row >= m_rows || col >= m_cols) return;
    
    size_t word_index = row * m_words_per_row + (col / 64);
    size_t bit_index = col % 64;
    
    if (value) {
        m_data[word_index] |= (1ULL << bit_index);
    } else {
        m_data[word_index] &= ~(1ULL << bit_index);
    }
}

void GF2Matrix::randomFill() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    
    for (auto& word : m_data) {
        word = dis(gen);
    }
}

GF2Matrix GF2Matrix::transpose() const {
    GF2Matrix result(m_cols, m_rows);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            if (get(i, j)) {
                result.set(j, i, true);
            }
        }
    }
    return result;
}

GF2Matrix GF2Matrix::multiplySerial(const GF2Matrix& other) const {
    if (m_cols != other.m_rows) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    GF2Matrix result(m_rows, other.m_cols);
    
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < other.m_cols; ++j) {
            bool sum = 0;
            for (size_t k = 0; k < m_cols; ++k) {
                bool a = get(i, k);
                bool b = other.get(k, j);
                sum ^= (a & b);
            }
            result.set(i, j, sum);
        }
    }
    
    return result;
}

// Forward declare the platform-specific SIMD functions
#if defined(__x86_64__) || defined(_M_X64)
void multiply_simd_x86(const GF2Matrix& a, const GF2Matrix& b, GF2Matrix& result);
#elif defined(__aarch64__)
void multiply_simd_neon(const GF2Matrix& a, const GF2Matrix& b, GF2Matrix& result);
#endif

GF2Matrix GF2Matrix::multiplySIMD(const GF2Matrix& other) const {
    if (m_cols != other.m_rows) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    GF2Matrix result(m_rows, other.m_cols);

#if defined(__x86_64__) || defined(_M_X64)
    // Dispatch to the AVX implementation on x86-64
    multiply_simd_x86(*this, other, result);
#elif defined(__aarch64__)
    // Dispatch to the NEON implementation on ARM64
    multiply_simd_neon(*this, other, result);
#else
    // Fallback for other architectures
    return multiplySerial(other);
#endif
    return result;
}

bool GF2Matrix::operator==(const GF2Matrix& other) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        return false;
    }
    
    // Compare row by row, ignoring padding
    for (size_t i = 0; i < m_rows; ++i) {
        if (memcmp(m_data.data() + i * m_words_per_row, other.m_data.data() + i * other.m_words_per_row, m_words_per_row * sizeof(uint64_t)) != 0) {
            return false;
        }
    }
    return true;
}

void GF2Matrix::print(size_t maxRows, size_t maxCols) const {
    size_t rowsToPrint = std::min(m_rows, maxRows);
    size_t colsToPrint = std::min(m_cols, maxCols);
    
    std::cout << "GF(2) Matrix " << m_rows << "x" << m_cols << ":\n";
    
    for (size_t i = 0; i < rowsToPrint; ++i) {
        for (size_t j = 0; j < colsToPrint; ++j) {
            std::cout << (get(i, j) ? "1" : "0");
            if (j < colsToPrint - 1) std::cout << " ";
        }
        if (colsToPrint < m_cols) {
            std::cout << " ...";
        }
        std::cout << "\n";
    }
    
    if (rowsToPrint < m_rows) {
        std::cout << "...\n";
    }
    std::cout << std::endl;
}

// popcnt64 and parity64 are not used in the current SIMD implementations
// but are kept as they were part of the original GF2Matrix.cpp
uint64_t GF2Matrix::popcnt64(uint64_t x) {
    // This is a placeholder; a real implementation would use __builtin_popcountll or equivalent
    // For GF(2) multiplication, we only care about parity, not the exact count.
    // The SIMD functions use __builtin_popcountll directly.
    uint64_t count = 0;
    while (x > 0) {
        x &= (x - 1);
        count++;
    }
    return count;
}

uint64_t GF2Matrix::parity64(uint64_t x) {
    // This is a placeholder; a real implementation would use __builtin_parityll or equivalent
    // For GF(2) multiplication, we only care about parity, not the exact count.
    // The SIMD functions use __builtin_popcountll % 2 directly.
    x ^= x >> 32;
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1;
}

