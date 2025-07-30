// --- File: gf2_multiply_m4r.metal ---
//
// Implements GF(2) matrix multiplication using the Method of Four Russians (M4R).
// This is a two-stage process:
// 1. Pre-computation: Build lookup tables from the B matrix.
// 2. Multiplication: Use the tables to accelerate the dot product calculation.
//
// This implementation is adapted from a CUDA version for the boolean semiring
// and has been modified to perform GF(2) arithmetic (XOR-AND) and operate on
// uint64_t packed data, consistent with the target framework.

#include <metal_stdlib>
using namespace metal;

// Use the same parameter struct as other kernels for consistency.
struct GPUParams {
    uint a_rows;
    uint a_cols; // The common dimension, K
    uint b_cols;
    uint words_per_row_a;
    uint words_per_row_b;
    uint words_per_row_result;
};

// --- M4R Algorithm Constants ---

// K_M4R: The number of bits used for a single table lookup key.
constant constexpr uint K_M4R = 8;

// TABLE_ROWS: The number of entries in each lookup table (2^K_M4R).
constant constexpr uint TABLE_ROWS = 1 << K_M4R; // 256

// --- Bit-Twiddling Helper Functions ---

// FIX: The 'function' keyword is not used for declaring standard helper functions in MSL.
// It is removed from the declarations below. The 'inline' keyword is a valid hint.

// Returns the least significant bit of an integer.
inline uint lsb(uint i) {
    return i & -i;
}

// "Snoob" (Stanford Nifty Organization or Bithacks) algorithm.
// Returns the next integer with the same number of set bits.
// This is used to efficiently build the lookup table by iterating through
// keys with 2 set bits, then 3, and so on.
inline uint snoob(uint i) {
    uint least = lsb(i);
    uint ripple = i + least;
    // The original CUDA code used `(ripple ^ i) >> 2) / least`.
    // This is equivalent to `((ripple ^ i) / least) >> 2`.
    // We use the latter for clarity, though compilers often optimize this.
    uint ones = ((ripple ^ i) / least) >> 2;
    return ripple | ones;
}


// --- Kernel 1: Table Generation ---
//
// This kernel builds the M4R lookup tables.
// We need one table for each 8-bit chunk of a row in matrix A.
// Total tables = words_per_row_a * (64 / K_M4R)
//
// Each thread computes one column of one lookup table.
// Grid dispatch: (words_per_row_b, words_per_row_a * 8)
kernel void m4r_make_tables_kernel(
    device const uint64_t* b [[buffer(0)]],
    device uint64_t* lookup_tables [[buffer(1)]],
    constant GPUParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint word_col_idx = gid.x; // Which word column of the table to compute.
    uint table_idx_flat = gid.y; // The flattened index of the table.

    // Boundary check.
    uint num_tables = params.words_per_row_a * (64 / K_M4R);
    if (word_col_idx >= params.words_per_row_b || table_idx_flat >= num_tables) {
        return;
    }

    // Deconstruct the flat table index to find which rows of B to use.
    uint table_word_idx = table_idx_flat / (64 / K_M4R); // Which word of A this table corresponds to.
    uint sub_table_idx = table_idx_flat % (64 / K_M4R);  // Which 8-bit chunk within that word.
    uint b_start_row = table_word_idx * 64 + sub_table_idx * K_M4R;

    // Calculate a pointer to the start of the column this thread is responsible for.
    // Each table has TABLE_ROWS entries, and each entry is words_per_row_b wide.
    device uint64_t* table_col_ptr = lookup_tables + (table_idx_flat * TABLE_ROWS * params.words_per_row_b) + word_col_idx;

    // The entry for key 0 is always zero.
    table_col_ptr[0] = 0;

    // Step 1: Fill base cases (keys with one bit set).
    // These correspond to single rows from matrix B.
    for (uint i = 0; i < K_M4R; ++i) {
        uint key = 1 << i;
        uint b_row_to_fetch = b_start_row + i;

        if (b_row_to_fetch < params.a_cols) {
            // The stride between entries in this column is `params.words_per_row_b`.
            table_col_ptr[key * params.words_per_row_b] = b[b_row_to_fetch * params.words_per_row_b + word_col_idx];
        } else {
            table_col_ptr[key * params.words_per_row_b] = 0;
        }
    }

    // Step 2: Fill the rest of the table using the snoob algorithm.
    // We compute entries for keys with 2 bits, then 3, etc., up to K_M4R.
    // Each entry is the XOR sum of two smaller entries.
    for (uint h = 2; h <= K_M4R; ++h) {
        uint i = (1 << h) - 1; // Start with the smallest number with h bits set.
        while (i < TABLE_ROWS) {
            uint least_bit = lsb(i);
            uint rest = i - least_bit;
            table_col_ptr[i * params.words_per_row_b] = table_col_ptr[least_bit * params.words_per_row_b] ^ table_col_ptr[rest * params.words_per_row_b];
            i = snoob(i);
        }
    }
}


// --- Kernel 2: Multiplication ---
//
// This kernel uses the pre-computed lookup tables to perform the multiplication.
// Each thread computes one uint64_t word of the result matrix C.
// Grid dispatch: (a_rows, words_per_row_result)
kernel void m4r_multiply_kernel(
    device const uint64_t* a [[buffer(0)]],
    device uint64_t* result [[buffer(1)]],
    device const uint64_t* lookup_tables [[buffer(2)]],
    constant GPUParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row_idx = gid.x;
    uint word_col_idx = gid.y; // Which word of the result row to compute.

    // Boundary check.
    if (row_idx >= params.a_rows || word_col_idx >= params.words_per_row_result) {
        return;
    }

    uint64_t result_word = 0;

    // Iterate over the words in the row of A.
    for (uint k_word = 0; k_word < params.words_per_row_a; ++k_word) {
        // Fetch a 64-bit word from matrix A.
        uint64_t a_word = a[row_idx * params.words_per_row_a + k_word];

        // The original CUDA code reversed bits. We preserve this behavior.
        // It affects how the 8-bit keys are extracted from the 64-bit word.
        a_word = reverse_bits(a_word);

        // Process the 64-bit word in 8-bit chunks (keys).
        for (uint j = 0; j < (64 / K_M4R); ++j) {
            // Extract the 8-bit key.
            uint key = (a_word >> (j * K_M4R)) & 0xFF;

            if (key == 0) {
                continue; // Key 0 contributes nothing.
            }

            // Find the correct lookup table.
            uint table_idx_flat = k_word * (64 / K_M4R) + j;

            // Pointer to the start of the correct table.
            device const uint64_t* table_ptr = lookup_tables + (table_idx_flat * TABLE_ROWS * params.words_per_row_b);

            // Pointer to the specific row in the table corresponding to our key.
            device const uint64_t* table_row_ptr = table_ptr + (key * params.words_per_row_b);

            // Fetch the pre-computed value and XOR it into our result.
            // We fetch the word at `word_col_idx`, which corresponds to the
            // column of the result matrix this thread is computing.
            result_word ^= table_row_ptr[word_col_idx];
        }
    }

    // Write the final computed word to the result matrix.
    result[row_idx * params.words_per_row_result + word_col_idx] = result_word;
}

