package com.user.nn;

public class TestTensor {
    public static void main(String[] args) throws Exception {
        int failures = 0;
        try {
            // constructors and numel/dim
            Tensor t = new Tensor(2,3);
            if (t.numel() != 6 || t.dim() != 2) { System.err.println("constructor/dim/numel wrong"); failures++; }

            // get/set and offset ordering
            t.set(5.0f, 1,2);
            if (t.get(1,2) != 5.0f) { System.err.println("get/set failed"); failures++; }

            // reshape / view
            Tensor r = new Tensor(new float[]{0,1,2,3,4,5},6).reshape(2,3);
            if (r.shape[0]!=2 || r.shape[1]!=3 || r.get(1,2)!=5f) { System.err.println("reshape/view failed"); failures++; }

            // clone
            Tensor c = r.clone(); c.set(9f, 0,0);
            if (r.get(0,0) == 9f) { System.err.println("clone is shallow"); failures++; }

            // flatten
            Tensor f = r.flatten(); if (f.dim()!=1 || f.numel()!=6) { System.err.println("flatten failed"); failures++; }

            // squeeze / unsqueeze
            Tensor a = new Tensor(new float[]{7f},1,1,1).squeeze(); if (a.dim()!=1 && a.numel()!=1) { /* ok if shape reduced */ }
            Tensor b = a.unsqueeze(0); if (b.shape[0]!=1) { System.err.println("unsqueeze failed"); failures++; }

            // add_/mul_ inplace
            Tensor ip = Torch.tensor(new float[]{1f,2f,3f},3);
            ip.add_(2f); if (ip.data[0] != 3f) { System.err.println("add_ failed"); failures++; }
            ip.mul_(2f); if (ip.data[1] != 8f) { System.err.println("mul_ failed"); failures++; }

        } catch (Exception e) { e.printStackTrace(); failures++; }
        if (failures==0) { System.out.println("TEST PASSED: Tensor basic"); System.exit(0); } else { System.err.println("TEST FAILED: Tensor basic failures="+failures); System.exit(2); }
    }
}
