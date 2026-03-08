package com.user.nn;

import com.user.nn.core.*;
import java.io.File;

public class TestSerialization {
    public static void main(String[] args) {
        System.out.println("Running TestSerialization...");
        boolean allPassed = true;
        allPassed &= testSaveLoadCPU();
        allPassed &= testSaveLoadGPU();

        if (allPassed) {
            System.out.println("TestSerialization PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestSerialization FAILED.");
            System.exit(1);
        }
    }

    private static boolean testSaveLoadCPU() {
        try {
            NN nn = new NN();
            NN.Sequential model = new NN.Sequential(
                new NN.Linear(nn, 10, 5, true),
                new NN.ReLU(),
                new NN.Linear(nn, 5, 2, true)
            );
            
            // Randomize and capture initial state
            float[] initialData = captureParams(model);
            
            String path = "test_model_cpu.bin";
            model.save(path);
            
            // Modify params slightly to ensure load actually does something
            NN.Parameter p = model.parameters().get(0);
            p.getTensor().data[0] += 1.0f;
            
            model.load(path);
            float[] loadedData = captureParams(model);
            
            check(isEqual(initialData, loadedData), "CPU Save/Load mismatch");
            
            new File(path).delete();
            System.out.println("  CPU Serialization OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testSaveLoadGPU() {
        try {
            if (!CUDAOps.isAvailable()) {
                System.out.println("  Skipping GPU Serialization (CUDA not available)");
                return true;
            }
            NN nn = new NN();
            NN.Sequential model = new NN.Sequential(
                new NN.Linear(nn, 10, 5, true),
                new NN.ReLU(),
                new NN.Linear(nn, 5, 2, true)
            );
            model.to(Tensor.Device.GPU);
            
            float[] initialData = captureParams(model);
            String path = "test_model_gpu.bin";
            model.save(path);
            
            model.load(path);
            float[] loadedData = captureParams(model);
            
            check(isEqual(initialData, loadedData), "GPU Save/Load mismatch");
            
            new File(path).delete();
            System.out.println("  GPU Serialization OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static float[] captureParams(NN.Module model) {
        int total = 0;
        for (NN.Parameter p : model.parameters()) {
            total += p.getTensor().numel();
        }
        float[] out = new float[total];
        int offset = 0;
        for (NN.Parameter p : model.parameters()) {
            Tensor t = p.getTensor();
            boolean wasGPU = t.isGPU();
            if (wasGPU) t.toCPU();
            System.arraycopy(t.data, 0, out, offset, t.data.length);
            offset += t.data.length;
            if (wasGPU) t.toGPU();
        }
        return out;
    }

    private static boolean isEqual(float[] a, float[] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 1e-6f) return false;
        }
        return true;
    }

    private static void check(boolean cond, String msg) {
        if (!cond) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }
}
