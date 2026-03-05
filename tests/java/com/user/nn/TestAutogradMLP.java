package com.user.nn;

public class TestAutogradMLP {
    public static void main(String[] args) {
        System.out.println("Running TestAutogradMLP...");

        boolean allPassed = true;
        allPassed &= testMLP();

        if (allPassed) {
            System.out.println("TestAutogradMLP PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestAutogradMLP FAILED.");
            System.exit(1);
        }
    }

    private static void check(boolean condition, String msg) {
        if (!condition) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static boolean testMLP() {
        try {
            nn outer = new nn();
            Torch.manual_seed(42);

            // Layer 1: 2 -> 4
            nn.Linear l1 = new nn.Linear(outer, 2, 4, true);
            // Layer 2: 4 -> 2
            nn.Linear l2 = new nn.Linear(outer, 4, 2, true);

            nn.Sequential model = new nn.Sequential();
            model.add(l1);
            model.add(new nn.ReLU());
            model.add(l2);

            // Forward pass test
            Tensor x = Torch.tensor(new float[] { 1.5f, -0.5f }, 1, 2);
            Tensor out = model.forward(x);

            // Expected output shape [1, 2]
            check(out.dim() == 2 && out.shape[0] == 1 && out.shape[1] == 2, "MLP output shape");

            int[] targets = new int[] { 1 };
            Tensor loss = nn.F.cross_entropy_tensor(out, targets);

            check(loss.dim() == 1 && loss.shape[0] == 1, "Loss shape");

            // Backward pass
            loss.backward();

            // Check gradients populated on weights and biases
            check(l1.weight.getTensor().grad != null, "L1 weight grad");
            check(l1.bias.getTensor().grad != null, "L1 bias grad");
            check(l2.weight.getTensor().grad != null, "L2 weight grad");
            check(l2.bias.getTensor().grad != null, "L2 bias grad");

            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
