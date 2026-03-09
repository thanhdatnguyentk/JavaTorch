package com.user.nn;
import com.user.nn.core.*;
import com.user.nn.optim.*;

public class TestAutogradMatmul {
    public static void main(String[] args) {
        System.out.println("Running TestAutogradMatmul...");

        boolean allPassed = true;
        allPassed &= testMatmul();
        allPassed &= testBmm();

        if (allPassed) {
            System.out.println("TestAutogradMatmul PASSED.");
            System.exit(0);
        } else {
            System.out.println("TestAutogradMatmul FAILED.");
            System.exit(1);
        }
    }

    private static void check(boolean condition, String msg) {
        if (!condition) {
            System.out.println("FAIL: " + msg);
            throw new RuntimeException("Assertion failed: " + msg);
        }
    }

    private static boolean testMatmul() {
        try {
            // A(2x3) * B(3x4) = C(2x4)
            Tensor a = Torch.tensor(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
            Tensor b = Torch.tensor(new float[] {
                    1, 1, 1, 1,
                    2, 2, 2, 2,
                    3, 3, 3, 3
            }, 3, 4);
            a.requires_grad = true;
            b.requires_grad = true;

            Tensor c = Torch.matmul(a, b);
            Tensor loss = Torch.sumTensor(c);
            loss.backward();

            check(a.grad != null && b.grad != null, "grads present");
            // Since loss = sum(C), every element in outGrad is 1.
            // dA = outGrad * B^T
            // dB = A^T * outGrad
            // B rows sum: b_0=4, b_1=8, b_2=12. So dA should be [4, 8, 12] repeated
            check(a.grad.data[0] == 4f && a.grad.data[1] == 8f && a.grad.data[2] == 12f, "dA pattern");
            // A cols sum: a_col0(1+4)=5, a_col1(2+5)=7, a_col2(3+6)=9. So dB should be [5,
            // 7, 9] mapped to rows
            check(b.grad.data[0] == 5f && b.grad.data[1] == 5f, "dB pattern row 0");
            check(b.grad.data[4] == 7f && b.grad.data[5] == 7f, "dB pattern row 1");
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean testBmm() {
        try {
            // A(2x1x2) batch matmul B(2x2x1) = C(2x1x1)
            Tensor a = Torch.tensor(new float[] {
                    1, 2, // batch 0
                    3, 4 // batch 1
            }, 2, 1, 2);
            Tensor b = Torch.tensor(new float[] {
                    1, // batch 0, row 0
                    2, // batch 0, row 1

                    0, // batch 1, row 0
                    1 // batch 1, row 1
            }, 2, 2, 1);

            a.requires_grad = true;
            b.requires_grad = true;

            Tensor c = Torch.bmm(a, b);
            Tensor loss = Torch.sumTensor(c);
            loss.backward();

            // C[0] = [1*1 + 2*2] = 5 -> grad scalar is 1
            // C[1] = [3*0 + 4*1] = 4 -> grad scalar is 1

            check(a.grad != null && b.grad != null, "grads present");
            // dA = outGrad * B^T. For batch 0: [1] * [1, 2] = [1, 2]
            check(a.grad.data[0] == 1f && a.grad.data[1] == 2f, "dA batch 0");
            check(a.grad.data[2] == 0f && a.grad.data[3] == 1f, "dA batch 1");

            // dB = A^T * outGrad. For batch 0: [1 \n 2] * [1] = [1, 2]
            check(b.grad.data[0] == 1f && b.grad.data[1] == 2f, "dB batch 0");
            check(b.grad.data[2] == 3f && b.grad.data[3] == 4f, "dB batch 1");

            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
