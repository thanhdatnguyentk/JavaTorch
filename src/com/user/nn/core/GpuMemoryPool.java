package com.user.nn.core;

import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.runtime.JCuda.*;

/**
 * A GPU Memory Pool (Arena Allocator) that pre-allocates a large contiguous block
 * of VRAM and distributes slices via pointer offsets, eliminating the overhead
 * of per-tensor cudaMalloc/cudaFree calls during training.
 *
 * Usage:
 *   GpuMemoryPool.init(2L * 1024 * 1024 * 1024);  // 2 GB
 *   Pointer slice = GpuMemoryPool.allocate(numElements);
 *   GpuMemoryPool.reset();  // Reset head pointer (instant "free all")
 */
public class GpuMemoryPool {
    private static Pointer poolBase = null;
    private static long poolSizeBytes = 0;
    private static long currentOffset = 0;
    private static boolean initialized = false;

    /**
     * Initialize the memory pool with a fixed size in bytes.
     * Should be called once at the start of training.
     */
    public static synchronized void init(long sizeBytes) {
        if (initialized) return;
        poolBase = new Pointer();
        cudaMalloc(poolBase, sizeBytes);
        poolSizeBytes = sizeBytes;
        currentOffset = 0;
        initialized = true;
        System.out.println("[GpuMemoryPool] Initialized with " + (sizeBytes / (1024 * 1024)) + " MB");
    }

    /**
     * Check if the pool has been initialized.
     */
    public static boolean isInitialized() {
        return initialized;
    }

    /**
     * Allocate a slice of GPU memory from the pool for the given number of float elements.
     * Returns a Pointer offset into the pre-allocated block.
     * Returns null if there is not enough space (caller should fall back to cudaMalloc).
     */
    public static synchronized Pointer allocate(int numFloats) {
        if (!initialized) return null;
        long bytesNeeded = (long) numFloats * Sizeof.FLOAT;
        // Align to 256 bytes for optimal GPU memory access
        bytesNeeded = ((bytesNeeded + 255) / 256) * 256;

        if (currentOffset + bytesNeeded > poolSizeBytes) {
            // Pool exhausted, caller should fall back to standard cudaMalloc
            return null;
        }

        Pointer slice = poolBase.withByteOffset(currentOffset);
        currentOffset += bytesNeeded;
        return slice;
    }

    /**
     * Reset the pool's head pointer back to zero.
     * This effectively "frees" all allocations instantly without calling cudaFree.
     * Should be called at the end of each training step (inside MemoryScope.close()).
     */
    public static synchronized void reset() {
        currentOffset = 0;
    }

    /**
     * Get the current usage of the pool in bytes.
     */
    public static long getUsedBytes() {
        return currentOffset;
    }

    /**
     * Get the total capacity of the pool in bytes.
     */
    public static long getCapacityBytes() {
        return poolSizeBytes;
    }

    /**
     * Destroy the pool and free the underlying VRAM.
     */
    public static synchronized void destroy() {
        if (initialized && poolBase != null) {
            cudaFree(poolBase);
            poolBase = null;
            poolSizeBytes = 0;
            currentOffset = 0;
            initialized = false;
        }
    }
}
