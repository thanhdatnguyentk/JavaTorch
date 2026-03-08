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
 *   GpuMemoryPool.autoInit();           // Auto-detect free VRAM, use 80%
 *   GpuMemoryPool.autoInit(0.6);        // Auto-detect free VRAM, use 60%
 *   GpuMemoryPool.init(2L * 1024 * 1024 * 1024);  // Manual: 2 GB
 *   Pointer slice = GpuMemoryPool.allocate(numElements);
 *   GpuMemoryPool.reset();  // Reset head pointer (instant "free all")
 */
public class GpuMemoryPool {
    private static Pointer poolBase = null;
    private static long poolSizeBytes = 0;
    private static long currentOffset = 0;
    private static boolean initialized = false;

    /**
     * Automatically initialize the memory pool by querying the GPU's free VRAM.
     * Allocates 80% of the available free memory by default.
     */
    public static synchronized void autoInit() {
        autoInit(0.8);
    }

    /**
     * Automatically initialize the memory pool by querying the GPU's free VRAM.
     * @param fraction The fraction of free VRAM to allocate (0.0 to 1.0).
     *                 For example, 0.8 means use 80% of free VRAM.
     */
    public static synchronized void autoInit(double fraction) {
        if (initialized) return;
        if (fraction <= 0 || fraction > 1.0) {
            throw new IllegalArgumentException("Fraction must be between 0 and 1.0, got: " + fraction);
        }
        
        long[] free = new long[1];
        long[] total = new long[1];
        cudaMemGetInfo(free, total);
        
        long allocBytes = (long) (free[0] * fraction);
        // Align to 1 MB boundary
        allocBytes = (allocBytes / (1024 * 1024)) * (1024 * 1024);
        
        System.out.printf("[GpuMemoryPool] GPU VRAM: Total=%d MB, Free=%d MB, Allocating=%.0f%% (%d MB)%n",
            total[0] / (1024 * 1024), free[0] / (1024 * 1024), fraction * 100, allocBytes / (1024 * 1024));
        
        init(allocBytes);
    }

    /**
     * Automatically initialize the memory pool based on the model's parameter count.
     * Computes the base memory required for parameters, gradients, and optimizer states,
     * then applies an overhead multiplier to account for forward activations and batch size.
     */
    public static synchronized void autoInit(Module model) {
        autoInit(model, 10.0f); // Default 10x multiplier for activations/workspace
    }

    /**
     * Automatically initialize the memory pool based on the model's parameter count.
     * @param model The neural network model.
     * @param overheadMultiplier Multiplier applied on base memory for activations (e.g. 5.0 to 15.0).
     */
    public static synchronized void autoInit(Module model, float overheadMultiplier) {
        if (initialized) return;
        
        long paramCount = model.countParameters();
        long paramBytes = paramCount * Sizeof.FLOAT;
        
        // Base footprint: Params + Gradients + Optimizer States (Adam has M and V) -> approx 4x param memory
        long baseBytes = paramBytes * 4; 
        
        long allocBytes = (long) (baseBytes * overheadMultiplier);
        
        // Minimum safety buffer (e.g. 512 MB)
        allocBytes = Math.max(allocBytes, 512L * 1024 * 1024);
        
        // Cap at 90% of available VRAM to prevent system freeze
        long[] free = new long[1];
        long[] total = new long[1];
        cudaMemGetInfo(free, total);
        
        long maxSafeBytes = (long) (free[0] * 0.9);
        allocBytes = Math.min(allocBytes, maxSafeBytes);
        
        // Align to 1 MB boundary
        allocBytes = (allocBytes / (1024 * 1024)) * (1024 * 1024);
        
        System.out.printf("[GpuMemoryPool] Model params: %,d. Base Memory: %d MB. Allocating: %d MB%n",
            paramCount, baseBytes / (1024 * 1024), allocBytes / (1024 * 1024));
            
        init(allocBytes);
    }

    /**
     * Initialize the memory pool with a fixed size in bytes.
     * Should be called once at the start of training.
     */
    public static synchronized void init(long sizeBytes) {
        if (initialized) return;
        poolBase = new Pointer();
        try {
            cudaMalloc(poolBase, sizeBytes);
        } catch (Exception e) {
            poolBase = null;
            System.err.println("[GpuMemoryPool] cudaMalloc failed for " + (sizeBytes / (1024 * 1024)) + " MB: " + e.getMessage());
            return;
        }
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
