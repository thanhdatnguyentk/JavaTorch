package com.user.nn;

public class TestContainers {
    public static void main(String[] args) {
        nn lib = new nn();
        // Test Sequential
        nn.Sequential seq = new nn.Sequential();
        nn.Linear l1 = new nn.Linear(lib, 4, 3, true);
        nn.ReLU r = new nn.ReLU();
        seq.add(l1);
        seq.add(r);

        // Input
        nn.Mat in = lib.mat_alloc(2, 4);
        lib.mat_fill(in, 1.0f);

        nn.Mat out = seq.forward(in);
        if (out.rows != 2 || out.cols != 3) {
            System.err.println("Sequential produced wrong shape");
            System.exit(1);
        }

        // Test ModuleList and ModuleDict basic API
        nn.ModuleList ml = new nn.ModuleList();
        ml.add(l1);
        try {
            ml.get(0);
        } catch (Exception e) {
            System.err.println("ModuleList get failed");
            System.exit(2);
        }

        nn.ModuleDict md = new nn.ModuleDict();
        md.put("layer", l1);
        if (md.getModule("layer") == null) { System.err.println("ModuleDict put/get failed"); System.exit(3); }

        System.out.println("TEST PASSED: Containers");
    }
}
