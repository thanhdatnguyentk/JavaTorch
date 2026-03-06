package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestGatherScatterExtras {
    public static void main(String[] args) throws Exception {
        int failures = 0;
        try {
            // negative indices for gather
            Tensor inp = Torch.tensor(new float[]{10,11,12,20,21,22},2,3);
            Tensor idxNeg = Torch.tensor(new float[]{-1,0,1, -2,-1,0},2,3);
            Tensor g = Torch.gather(inp, 1, idxNeg);
            // -1 -> last column (2, 5)
            if (g.data[0] != 12f || g.data[3] != 21f) { System.err.println("gather negative failed: " + java.util.Arrays.toString(g.data)); failures++; }

            // scatter_ in-place
            Tensor base = Torch.zeros(2,3);
            Tensor index = Torch.tensor(new float[]{0,1,2,0,1,2},2,3);
            Tensor src = Torch.tensor(new float[]{1,2,3,4,5,6},2,3);
            Torch.scatter_(base, 1, index, src);
            if (base.data[0] != 1f || base.data[5] != 6f) { System.err.println("scatter_ failed: " + java.util.Arrays.toString(base.data)); failures++; }

        } catch (Exception e) { e.printStackTrace(); failures++; }
        if (failures==0) { System.out.println("TEST PASSED: Gather/Scatter extras"); System.exit(0); } else { System.err.println("TEST FAILED: Gather/Scatter extras failures="+failures); System.exit(2); }
    }
}
