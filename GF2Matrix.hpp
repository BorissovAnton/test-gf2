#pragma once

#include <vector>
#include <cstdint>
#include <random>

class GF2Matrix {
public:
    // Constructor
    GF2Matrix(size_t rows, size_t cols);
    
    // Accessors
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    size_t words_per_row() const { return m_words_per_row; }
    const uint64_t* get_raw_data() const { return m_data.data(); }
    
    // Get/set bit at position (row, col)
    bool get(size_t row, size_t col) const;
    void set(size_t row, size_t col, bool value);
    
    // Fill with random bits
    void randomFill();
    
    // Matrix multiplication (serial implementation)
    GF2Matrix multiplySerial(const GF2Matrix& other) const;
    
    // Matrix multiplication (SIMD implementation)
    GF2Matrix multiplySIMD(const GF2Matrix& other) const;

    // Transpose the matrix
    GF2Matrix transpose() const;
    
    // Matrix comparison
    bool operator==(const GF2Matrix& other) const;
    
    // Debug output
    void print(size_t maxRows = 10, size_t maxCols = 10) const;
    
private:
    size_t m_rows;
    size_t m_cols;
    size_t m_words_per_row;
    std::vector<uint64_t> m_data;
    
    // Internal multiplication helpers
    static uint64_t popcnt64(uint64_t x);
    static uint64_t parity64(uint64_t x); // This function is not used in the provided code, but kept for completeness if it was intended.
};

// GPU kernel interface
#ifdef __METAL_VERSION__
kernel void gf2_multiply(
    device const uint64_t* a,
    device const uint64_t* b,
    device uint64_t* result,
    constant uint* params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
);
#endif

