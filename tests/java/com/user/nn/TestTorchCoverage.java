package com.user.nn;

public class TestTorchCoverage {
    public static void main(String[] args) throws Exception {
        int failures = 0;
        try {
            // creation
            Tensor z = Torch.zeros(2,3);
            Tensor o = Torch.ones(2,3);
            Tensor r = Torch.arange(0,6).reshape(2,3);
            Tensor l = Torch.linspace(0f,1f,5);
            Tensor id = Torch.eye(3);
            if (z.numel()!=6 || o.numel()!=6) { System.err.println("zeros/ones size mismatch"); failures++; }

            // math
            Tensor s = Torch.add(r, 1.0f);
            Tensor p = Torch.mul(s, 2.0f);
            if (p.data[0] != (r.data[0]+1f)*2f) { System.err.println("add/mul error"); failures++; }

            // broadcast
            Tensor a = new Tensor(new float[]{1,2,3},3);
            Tensor b = Torch.full(new int[]{3,3}, 2f);
            Tensor c = Torch.add(a.reshape(3,1), b); // broadcast

            // reductions
            float sum = Torch.sum(c);
            if (sum==0f) { System.err.println("sum seems zero"); failures++; }

            // matmul
            Tensor A = Torch.tensor(new float[]{1,2,3,4},2,2);
            Tensor B = Torch.tensor(new float[]{5,6,7,8},2,2);
            Tensor M = Torch.matmul(A,B);
            if (M.shape[0]!=2 || M.shape[1]!=2) { System.err.println("matmul shape"); failures++; }

            // rand/int
            Tensor rnd = Torch.randint(0,10,3,3);

            // save/load
            String path = "tests/tmp/tensor_save.txt";
            Torch.save(rnd, path);
            Tensor rnd2 = Torch.load(path);
            if (rnd2.numel() != rnd.numel()) { System.err.println("save/load mismatch"); failures++; }

            // grad control
            Torch.enable_grad();
            if (!Torch.is_grad_enabled()) { System.err.println("enable_grad failed"); failures++; }
            try (java.io.Closeable cno = Torch.no_grad()) {
                if (Torch.is_grad_enabled()) { System.err.println("no_grad did not disable"); failures++; }
            }
            if (!Torch.is_grad_enabled()) { System.err.println("no_grad did not restore"); failures++; }

        } catch (Exception e) {
            e.printStackTrace(); failures++;
        }
        if (failures==0) { System.out.println("TEST PASSED: Torch coverage"); System.exit(0); } else { System.err.println("TEST FAILED: Torch coverage failures="+failures); System.exit(2); }
    }
}
