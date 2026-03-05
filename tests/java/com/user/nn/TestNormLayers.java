package com.user.nn;

/**
 * Tests for LayerNorm and InstanceNorm with autograd backward.
 */
public class TestNormLayers {
    public static void main(String[] args) {
        System.out.println("Running TestNormLayers...");
        boolean allPassed = true;
        allPassed &= testLayerNorm();
        allPassed &= testInstanceNorm();

        if (allPassed) {
            System.out.println("TestNormLayers PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestNormLayers FAILED.");
            System.exit(1);
        }
    }

    private static void check(boolean cond, String msg) {
        if (!cond) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static boolean testLayerNorm() {
        try {
            nn outer = new nn();
            int D = 4;
            nn.LayerNorm ln = new nn.LayerNorm(outer, D);

            // batch=2, features=4
            Tensor x = Torch.tensor(new float[] {
                    1f, 2f, 3f, 4f, // sample 0: mean=2.5, var=1.25
                    4f, 4f, 4f, 4f // sample 1: all same => normalized=0
            }, 2, D);
            x.requires_grad = true;

            Tensor out = ln.forward(x);

            // Check sample 1: all values should be ~0 (since all inputs are the same)
            for (int d = 0; d < D; d++) {
                check(Math.abs(out.data[1 * D + d]) < 1e-4f,
                        "LayerNorm sample1 out[" + d + "] near 0, got " + out.data[1 * D + d]);
            }

            // Check sample 0: mean of normalized should be ~0
            float mean = 0f;
            for (int d = 0; d < D; d++)
                mean += out.data[d];
            mean /= D;
            check(Math.abs(mean) < 1e-4f, "LayerNorm sample0 mean near 0, got " + mean);

            // backward
            Tensor loss = Torch.sumTensor(out);
            loss.backward();
            check(x.grad != null, "LayerNorm input grad exists");
            check(ln.weight.getTensor().grad != null, "LayerNorm gamma grad exists");
            check(ln.bias.getTensor().grad != null, "LayerNorm beta grad exists");

            // beta grad: each beta dimension gets sum over batch of 1 = 2
            for (int d = 0; d < D; d++) {
                check(Math.abs(ln.bias.getTensor().grad.data[d] - 2f) < 1e-4f,
                        "LayerNorm beta grad[" + d + "]");
            }

            System.out.println("  LayerNorm OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testInstanceNorm() {
        try {
            int C = 2, H = 2, W = 2;
            nn.InstanceNorm in_ = new nn.InstanceNorm(C, H, W);

            // batch=1, C=2, H=2, W=2 => flattened size = 8
            // channel 0: [1,2,3,4] => mean=2.5, normalized non-zero
            // channel 1: [5,5,5,5] => mean=5, normalized=0
            Tensor x = Torch.tensor(new float[] { 1, 2, 3, 4, 5, 5, 5, 5 }, 1, C * H * W);
            x.requires_grad = true;

            Tensor out = in_.forward(x);

            // Check channel 1: all zeros
            for (int hw = 0; hw < H * W; hw++) {
                check(Math.abs(out.data[C * H * W * 0 + 1 * H * W + hw]) < 1e-4f,
                        "InstanceNorm channel1 out[" + hw + "] near 0");
            }

            // Check channel 0: mean ~0
            float mean = 0f;
            for (int hw = 0; hw < H * W; hw++)
                mean += out.data[hw];
            mean /= (H * W);
            check(Math.abs(mean) < 1e-4f, "InstanceNorm channel0 mean near 0, got " + mean);

            // backward
            Tensor loss = Torch.sumTensor(out);
            loss.backward();
            check(x.grad != null, "InstanceNorm input grad exists");

            System.out.println("  InstanceNorm OK");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
