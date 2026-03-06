package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestContainers {
    public static void main(String[] args) {
        NN lib = new NN();
        // Test Sequential
        NN.Sequential seq = new NN.Sequential();
        NN.Linear l1 = new NN.Linear(lib, 4, 3, true);
        NN.ReLU r = new NN.ReLU();
        seq.add(l1);
        seq.add(r);

        // Input
        NN.Mat in = lib.mat_alloc(2, 4);
        lib.mat_fill(in, 1.0f);

        NN.Mat out = seq.forward(in);
        if (out.rows != 2 || out.cols != 3) {
            System.err.println("Sequential produced wrong shape");
            System.exit(1);
        }

        // Test ModuleList and ModuleDict basic API
        NN.ModuleList ml = new NN.ModuleList();
        ml.add(l1);
        try {
            ml.get(0);
        } catch (Exception e) {
            System.err.println("ModuleList get failed");
            System.exit(2);
        }

        NN.ModuleDict md = new NN.ModuleDict();
        md.put("layer", l1);
        if (md.getModule("layer") == null) { System.err.println("ModuleDict put/get failed"); System.exit(3); }

        System.out.println("TEST PASSED: Containers");
    }
}
