package com.user.nn;

import jcuda.jcudnn.*;
import static jcuda.jcudnn.JCudnn.*;

public class TestCudnn {
    public static void main(String[] args) {
        System.out.println("Testing cuDNN Initialization...");
        try {
            cudnnHandle handle = new cudnnHandle();
            int result = cudnnCreate(handle);
            if (result == cudnnStatus.CUDNN_STATUS_SUCCESS) {
                System.out.println("cuDNN Initialized Successfully!");
                System.out.println("cuDNN Version: " + cudnnGetVersion());
                cudnnDestroy(handle);
            } else {
                System.err.println("Failed to initialize cuDNN: " + cudnnGetErrorString(result));
            }
        } catch (Throwable t) {
            System.err.println("Error loading cuDNN native library or similar:");
            t.printStackTrace();
        }
    }
}
