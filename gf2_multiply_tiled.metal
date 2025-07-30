// --- File: gf2_multiply_tiled.metal ---

#include <metal_stdlib>
using namespace metal;

// Use the same parameters struct as the original kernel
struct GF2Params {
    uint a_rows;
    uint a_cols;
    uint b_cols;
    uint words_per_row_a;
    uint words_per_row_b;
    uint words_per_row_result;
};

// --- Bit Manipulation Helpers ---

// FIX: Removed the boundary check from inside the helper. The calling kernel
// already performs a more accurate check before calling this function.
inline uint get_bit(device const uint64_t* data, uint row, uint col, uint words_per_row) {
    uint word_idx = col / 64;
    uint bit_idx = col % 64;
    uint64_t word = data[row * words_per_row + word_idx];
    return (word >> bit_idx) & 1ULL;
}

// FIX: Corrected the latent type mismatch bug. The result buffer is atomic_uint (32-bit),
// so the mask must also be a 32-bit uint.
inline void set_bit_atomic(volatile device atomic_uint* data, uint row, uint col, uint words_per_row) {
    uint word_idx = col / 64;
    uint bit_idx = col % 64;
    // The mask must be 'uint' to match the 'atomic_uint' base type.
    uint mask = 1U << bit_idx;
    atomic_fetch_or_explicit(&data[row * words_per_row + word_idx], mask, memory_order_relaxed);
}


// --- Tiled Multiplication Kernel ---

#define TILE_WIDTH 32

kernel void gf2_multiply_tiled_kernel(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b [[buffer(1)]],
    volatile device atomic_uint* result [[buffer(2)]],
    constant GF2Params& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    // FIX: Use the correct attribute for a 2D vector thread coordinate.
    uint2 tid_in_group [[thread_position_in_threadgroup]])
{
    // Global position of the bit this thread will compute
    uint row = gid.x;
    uint col = gid.y;

    // Local position of the thread within its 32x32 threadgroup
    uint local_row = tid_in_group.x;
    uint local_col = tid_in_group.y;

    // Declare fast on-chip threadgroup (shared) memory
    threadgroup uint a_tile[TILE_WIDTH][TILE_WIDTH];
    threadgroup uint b_tile[TILE_WIDTH][TILE_WIDTH];

    uint sum = 0;
    uint num_tiles = (params.a_cols + TILE_WIDTH - 1) / TILE_WIDTH;

    // Loop over the tiles along the common dimension 'k'
    for (uint tile_k = 0; tile_k < num_tiles; ++tile_k) {
        // Calculate the global memory coordinates for this tile
        uint tile_k_offset = tile_k * TILE_WIDTH;
        uint a_col_to_load = tile_k_offset + local_col;
        uint b_row_to_load = tile_k_offset + local_row;

        // Load one bit into the A tile. This is a COALESCED read.
        if (row < params.a_rows && a_col_to_load < params.a_cols) {
            a_tile[local_row][local_col] = get_bit(a, row, a_col_to_load, params.words_per_row_a);
        } else {
            a_tile[local_row][local_col] = 0;
        }

        // Load one bit into the B tile. This is a STRIDED read.
        // Perform an on-the-fly transpose: store at [local_col][local_row].
        if (b_row_to_load < params.a_cols && col < params.b_cols) {
            b_tile[local_col][local_row] = get_bit(b, b_row_to_load, col, params.words_per_row_b);
        } else {
            b_tile[local_col][local_row] = 0;
        }

        // Synchronize all threads in the group to ensure tiles are fully loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product from fast threadgroup memory
        for (uint k_local = 0; k_local < TILE_WIDTH; ++k_local) {
            sum ^= a_tile[local_row][k_local] & b_tile[k_local][local_col];
        }

        // Synchronize again before the next iteration loads new data
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // After all tiles are processed, write the final result to global memory if non-zero.
    if (sum && row < params.a_rows && col < params.b_cols) {
        set_bit_atomic(result, row, col, params.words_per_row_result);
    }
}

