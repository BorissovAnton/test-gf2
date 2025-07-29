#include "GF2Matrix.hpp"
#include <iostream>
#include <cstring>
// #include <immintrin.h> 

GF2Matrix::GF2Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
    size_t totalBits = rows * cols;
    size_t dataSize = (totalBits + 63) / 64;
    m_data.resize(dataSize, 0);
}

bool GF2Matrix::get(size_t row, size_t col) const {
    if (row >= m_rows || col >= m_cols) return false;
    
    size_t index = row * m_cols + col;
    size_t dataIndex = index / 64;
    size_t bitIndex = index % 64;
    
    return (m_data[dataIndex] >> bitIndex) & 1ULL;
}

void GF2Matrix::set(size_t row, size_t col, bool value) {
    if (row >= m_rows || col >= m_cols) return;
    
    size_t index = row * m_cols + col;
    size_t dataIndex = index / 64;
    size_t bitIndex = index % 64;
    
    if (value) {
        m_data[dataIndex] |= (1ULL << bitIndex);
    } else {
        m_data[dataIndex] &= ~(1ULL << bitIndex);
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

GF2Matrix GF2Matrix::multiplySIMD(const GF2Matrix& other) const {
    if (m_cols != other.m_rows) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    GF2Matrix result(m_rows, other.m_cols);
    
    // For now, fall back to serial implementation as a placeholder
    // Real SIMD implementation would use AVX2/AVX-512 bit manipulation
    return multiplySerial(other);
}

bool GF2Matrix::operator==(const GF2Matrix& other) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        return false;
    }
    
    return m_data == other.m_data;
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

size_t GF2Matrix::dataIndex(size_t row, size_t col) const {
    size_t index = row * m_cols + col;
    return index / 64;
}

size_t GF2Matrix::bitIndex(size_t row, size_t col) const {
    size_t index = row * m_cols + col;
    return index % 64;
}

