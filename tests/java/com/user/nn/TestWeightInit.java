package com.user.nn;

import com.user.nn.core.*;

public class TestWeightInit {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; }
        else { failed++; System.out.println("FAIL: " + name); }
    }

    static float mean(float[] d) {
        double s = 0; for (float v : d) s += v; return (float)(s / d.length);
    }
    static float variance(float[] d, float mean) {
        double s = 0; for (float v : d) s += (v - mean) * (v - mean); return (float)(s / d.length);
    }

    public static void main(String[] args) {
        Torch.manual_seed(42);

        // --- uniform_ ---
        Tensor t1 = new Tensor(1000);
        Torch.nn.init.uniform_(t1, -1f, 1f);
        float m1 = mean(t1.data);
        check("uniform_ mean near 0", Math.abs(m1) < 0.1f);
        check("uniform_ range", t1.data[0] >= -1f && t1.data[0] <= 1f);

        // --- normal_ ---
        Tensor t2 = new Tensor(10000);
        Torch.nn.init.normal_(t2, 0f, 1f);
        float m2 = mean(t2.data);
        float v2 = variance(t2.data, m2);
        check("normal_ mean near 0", Math.abs(m2) < 0.05f);
        check("normal_ var near 1", Math.abs(v2 - 1.0f) < 0.1f);

        // --- zeros_, ones_, constant_ ---
        Tensor t3 = new Tensor(100);
        Torch.nn.init.zeros_(t3);
        check("zeros_", t3.data[0] == 0f && t3.data[99] == 0f);

        Tensor t4 = new Tensor(100);
        Torch.nn.init.ones_(t4);
        check("ones_", t4.data[0] == 1f && t4.data[99] == 1f);

        Tensor t5 = new Tensor(100);
        Torch.nn.init.constant_(t5, 3.14f);
        check("constant_", Math.abs(t5.data[0] - 3.14f) < 1e-6f);

        // --- xavier_uniform_ for 2D [out=256, in=512] ---
        Tensor t6 = new Tensor(256, 512);
        Torch.nn.init.xavier_uniform_(t6);
        float m6 = mean(t6.data);
        float v6 = variance(t6.data, m6);
        // Expected: U[-a, a] where a = sqrt(6/(512+256)) = sqrt(6/768) ≈ 0.0884
        // Variance of U[-a,a] = a^2/3 = 6/(3*768) = 2/768 ≈ 0.0026
        float expectedVar6 = 2.0f / (512 + 256);
        check("xavier_uniform_ mean near 0", Math.abs(m6) < 0.01f);
        check("xavier_uniform_ variance", Math.abs(v6 - expectedVar6) < 0.001f);

        // --- xavier_normal_ ---
        Tensor t7 = new Tensor(256, 512);
        Torch.nn.init.xavier_normal_(t7);
        float m7 = mean(t7.data);
        float v7 = variance(t7.data, m7);
        // Expected: std = sqrt(2/(512+256)) = sqrt(2/768), var = 2/768 ≈ 0.0026
        float expectedVar7 = 2.0f / (512 + 256);
        check("xavier_normal_ mean near 0", Math.abs(m7) < 0.01f);
        check("xavier_normal_ variance", Math.abs(v7 - expectedVar7) < 0.001f);

        // --- kaiming_uniform_ for 2D [out=256, in=512] ---
        Tensor t8 = new Tensor(256, 512);
        Torch.nn.init.kaiming_uniform_(t8);
        float m8 = mean(t8.data);
        float v8 = variance(t8.data, m8);
        // gain = sqrt(2) for relu; std = gain / sqrt(fan_in) = sqrt(2)/sqrt(512); bound = sqrt(3)*std
        // var of U[-b,b] = b^2/3 = 3*std^2/3 = std^2 = 2/512 ≈ 0.0039
        float expectedVar8 = 2.0f / 512;
        check("kaiming_uniform_ mean near 0", Math.abs(m8) < 0.01f);
        check("kaiming_uniform_ variance", Math.abs(v8 - expectedVar8) < 0.001f);

        // --- kaiming_normal_ ---
        Tensor t9 = new Tensor(256, 512);
        Torch.nn.init.kaiming_normal_(t9);
        float m9 = mean(t9.data);
        float v9 = variance(t9.data, m9);
        float expectedVar9 = 2.0f / 512;
        check("kaiming_normal_ mean near 0", Math.abs(m9) < 0.01f);
        check("kaiming_normal_ variance", Math.abs(v9 - expectedVar9) < 0.001f);

        // --- fan calculation for 4D (conv) ---
        int[] fan = Torch.nn.init.calculateFanInOut(new Tensor(64, 32, 3, 3));
        check("fan_in_4D", fan[0] == 32 * 3 * 3); // 288
        check("fan_out_4D", fan[1] == 64 * 3 * 3); // 576

        // --- gain calculations ---
        check("gain_relu", Math.abs(Torch.nn.init.calculateGain("relu") - Math.sqrt(2.0)) < 1e-5f);
        check("gain_tanh", Math.abs(Torch.nn.init.calculateGain("tanh") - 5.0f/3.0f) < 1e-5f);
        check("gain_linear", Torch.nn.init.calculateGain("linear") == 1.0f);

        System.out.println("TestWeightInit: " + passed + " passed, " + failed + " failed.");
        if (failed > 0) System.exit(1);
    }
}
