package com.user.nn.core;

/**
 * Global configuration for Mixed Precision (FP16) training.
 * When enabled, cuBLAS and cuDNN will use Tensor Cores internally
 * to accelerate matmul and convolution operations.
 * 
 * Usage:
 *   MixedPrecision.enable();   // Turn on Tensor Core acceleration
 *   MixedPrecision.disable();  // Turn off (default)
 *   MixedPrecision.isEnabled(); // Check status
 *
 * Note: All data stays as float32 in Java. The GPU handles
 * FP16 conversion internally for Tensor Core paths.
 */
public class MixedPrecision {
    private static boolean enabled = false;

    /** Enable Mixed Precision — activates Tensor Core (FP16) acceleration. */
    public static void enable() {
        enabled = true;
        System.out.println("[MixedPrecision] Enabled — Tensor Cores (FP16) will be used for matmul and convolution.");
    }

    /** Disable Mixed Precision — reverts to standard FP32 compute. */
    public static void disable() {
        enabled = false;
        System.out.println("[MixedPrecision] Disabled — using standard FP32 compute.");
    }

    /** Check if Mixed Precision is currently enabled. */
    public static boolean isEnabled() {
        return enabled;
    }
}
