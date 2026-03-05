package com.user.nn;

import java.util.ArrayList;
import java.util.List;

public class TestTorchExtras {
    public static void main(String[] args) throws Exception {
        int failures = 0;
        try {
            // stack / cat / split / chunk
            Tensor t1 = Torch.tensor(new float[]{1,2},2);
            Tensor t2 = Torch.tensor(new float[]{3,4},2);
            List<Tensor> lst = new ArrayList<>(); lst.add(t1); lst.add(t2);
            Tensor stacked = Torch.stack(lst, 0); // shape (2,2)
            if (stacked.shape.length!=2 || stacked.shape[0]!=2 || stacked.shape[1]!=2) { System.err.println("stack shape wrong"); failures++; }

            Tensor seq = Torch.arange(0,6).reshape(6);
            List<Tensor> parts = Torch.split(seq, new int[]{2,2,2}, 0);
            if (parts.size()!=3 || parts.get(0).numel()!=2) { System.err.println("split failed"); failures++; }
            List<Tensor> chunks = Torch.chunk(seq, 3, 0);
            if (chunks.size()!=3) { System.err.println("chunk failed"); failures++; }

            // where
            Tensor cond = Torch.tensor(new float[]{1,0,1,0},4);
            Tensor xa = Torch.tensor(new float[]{10,11,12,13},4);
            Tensor ya = Torch.tensor(new float[]{100,101,102,103},4);
            Tensor w = Torch.where(cond, xa, ya);
            if (w.data[0]!=10f || w.data[1]!=101f) { System.err.println("where incorrect"); failures++; }

            // permute
            Tensor A = Torch.arange(0,6).reshape(2,3);
            Tensor At = Torch.permute(A, 1,0);
            if (At.shape[0]!=3 || At.shape[1]!=2 || At.data[0] != A.data[0]) { System.err.println("permute wrong"); failures++; }

            // gather/scatter
            Tensor inp = Torch.tensor(new float[]{10,11,12,20,21,22},2,3);
            Tensor idx = Torch.tensor(new float[]{2,0,1,1,2,0},2,3);
            Tensor g = Torch.gather(inp, 1, idx);
            if (g.data[0] != 12f || g.data[3] != 21f) { System.err.println("gather wrong"); failures++; }
            Tensor zeros = Torch.zeros(2,3);
            Tensor s = Torch.scatter(zeros, 1, idx, Torch.tensor(new float[]{1,2,3,4,5,6},2,3));
            // expected resulting layout (row-major): [2,3,1,6,4,5]
            if (s.data[0] != 2f || s.data[5] != 5f) { System.err.println("scatter wrong: " + java.util.Arrays.toString(s.data)); failures++; }

            // trig/log/trunc/comparisons
            Tensor v = Torch.tensor(new float[]{0.5f, -0.5f},2);
            Tensor as = Torch.asin(v); Tensor ac = Torch.acos(v); Tensor at = Torch.atan(v);
            Tensor l10 = Torch.log10(Torch.tensor(new float[]{10f},1)); if (Math.abs(l10.data[0]-1.0f) > 1e-6) { System.err.println("log10 wrong"); failures++; }
            Tensor l2 = Torch.log2(Torch.tensor(new float[]{8f},1)); if (Math.abs(l2.data[0]-3.0f) > 1e-6) { System.err.println("log2 wrong"); failures++; }
            Tensor tr = Torch.trunc(Torch.tensor(new float[]{1.9f, -1.9f},2)); if (tr.data[0]!=1f || tr.data[1]!=-1f) { System.err.println("trunc wrong"); failures++; }
            Tensor cmp = Torch.ge(Torch.tensor(new float[]{1,2},2), Torch.tensor(new float[]{1,3},2)); if (cmp.data[0]!=1f || cmp.data[1]!=0f) { System.err.println("ge wrong"); failures++; }

            // reductions: var/std/argmin/norm
            Tensor numbers = Torch.tensor(new float[]{1f,2f,3f,4f},4);
            float var = Torch.var(numbers); if (Math.abs(var - 1.25f) > 1e-6) { System.err.println("var wrong: " + var); failures++; }
            float std = Torch.std(numbers); if (Math.abs(std - (float)Math.sqrt(1.25)) > 1e-6) { System.err.println("std wrong"); failures++; }
            Tensor mat2 = Torch.tensor(new float[]{5f,2f,3f,1f,4f,6f},2,3);
            int[] amin = Torch.argmin(mat2, 1); if (amin[0]!=1 || amin[1]!=0) { System.err.println("argmin wrong"); failures++; }
            Tensor v2 = Torch.tensor(new float[]{3f,4f},2); float nrm = Torch.norm(v2); if (Math.abs(nrm - 5f) > 1e-6) { System.err.println("norm wrong"); failures++; }

            // inverse/det
            Tensor M = Torch.tensor(new float[]{1f,2f,3f,4f},2,2);
            float d = Torch.det(M); if (Math.abs(d - (-2f)) > 1e-6) { System.err.println("det wrong: " + d); failures++; }
            Tensor Minv = Torch.inverse(M);
            if (Math.abs(Minv.data[0] - (-2f)) > 1e-6) { System.err.println("inverse wrong"); failures++; }

            // bernoulli / multinomial
            Tensor b0 = Torch.bernoulli(0.0f, 10); for (int i=0;i<b0.numel();i++) if (b0.data[i]!=0f) { System.err.println("bernoulli 0 failed"); failures++; }
            Tensor b1 = Torch.bernoulli(1.0f, 5); for (int i=0;i<b1.numel();i++) if (b1.data[i]!=1f) { System.err.println("bernoulli 1 failed"); failures++; }
            Tensor probs = Torch.tensor(new float[]{0f,1f,0f},3);
            Tensor m = Torch.multinomial(probs, 3, true);
            for (int i=0;i<m.numel();i++) if (m.data[i] != 1f) { System.err.println("multinomial failed"); failures++; }

        } catch (Exception e) {
            e.printStackTrace(); failures++;
        }
        if (failures==0) { System.out.println("TEST PASSED: Torch extras"); System.exit(0); } else { System.err.println("TEST FAILED: Torch extras failures="+failures); System.exit(2); }
    }
}
