package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.containers.*;
import com.user.nn.activations.*;

public class TestContainers {
    public static void main(String[] args) {
        // Test Sequential
        Sequential seq = new Sequential();
        Linear l1 = new Linear(4, 3, true);
        ReLU r = new ReLU();
        seq.add(l1);
        seq.add(r);

        // Input
        NN.Mat in = NN.mat_alloc(2, 4);
        NN.mat_fill(in, 1.0f);

        NN.Mat out = seq.forward(in);
        if (out.rows != 2 || out.cols != 3) {
            System.err.println("Sequential produced wrong shape");
            System.exit(1);
        }

        // Test ModuleList and ModuleDict basic API
        ModuleList ml = new ModuleList();
        ml.add(l1);
        try {
            ml.get(0);
        } catch (Exception e) {
            System.err.println("ModuleList get failed");
            System.exit(2);
        }

        ModuleDict md = new ModuleDict();
        md.put("layer", l1);
        if (md.getModule("layer") == null) { System.err.println("ModuleDict put/get failed"); System.exit(3); }

        System.out.println("TEST PASSED: Containers");
    }
}
