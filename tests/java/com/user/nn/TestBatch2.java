package com.user.nn;

public class TestBatch2 {
    public static void main(String[] args) {
        testL1Loss();
        testBCELoss();
        testBCEWithLogitsLoss();
        testKLDivLoss();
        testCosineSimilarity();
        testPairwiseDistance();
        System.out.println("All Batch 2 tests PASSED!");
    }

    private static void check(String name, boolean cond) {
        if (cond) {
            System.out.println("  PASS: " + name);
        } else {
            System.err.println("  FAIL: " + name);
            System.exit(1);
        }
    }

    private static void testL1Loss() {
        System.out.println("Testing L1Loss...");
        Tensor pred = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        pred.requires_grad = true;
        Tensor target = new Tensor(new float[]{1.5f, 1.5f, 3.5f}, 3);
        Tensor loss = nn.F.l1_loss(pred, target);
        // |1-1.5| + |2-1.5| + |3-3.5| = 0.5 + 0.5 + 0.5 = 1.5. mean = 1.5/3 = 0.5
        check("L1Loss value", Math.abs(loss.data[0] - 0.5f) < 1e-6f);

        loss.backward();
        // dL/dx = sign(pred-target)/n = [-1, 1, -1] / 3
        check("L1Loss grad[0]", Math.abs(pred.grad.data[0] - (-1f/3f)) < 1e-6f);
        check("L1Loss grad[1]", Math.abs(pred.grad.data[1] - (1f/3f)) < 1e-6f);
    }

    private static void testBCELoss() {
        System.out.println("Testing BCELoss...");
        Tensor pred = new Tensor(new float[]{0.1f, 0.9f}, 2);
        pred.requires_grad = true;
        Tensor target = new Tensor(new float[]{0.0f, 1.0f}, 2);
        Tensor loss = nn.F.binary_cross_entropy(pred, target);
        // Loss = -(0*log(0.1) + 1*log(0.9) + 1*log(0.9) + 0*log(0.1)) ? No, reduction=mean
        // sample 0: -(0*log(0.1) + 1*log(0.9)) = -log(0.9) = 0.10536
        // sample 1: -(1*log(0.9) + 0*log(0.1)) = -log(0.9) = 0.10536
        // mean ≈ 0.10536
        check("BCELoss value", Math.abs(loss.data[0] - 0.10536f) < 1e-4f);

        loss.backward();
        // grad = (pred - target) / (pred * (1-pred) * n)
        // [0.1-0]/[0.1*0.9*2] = 0.1 / 0.18 = 0.55555...
        // [0.9-1]/[0.9*0.1*2] = -0.1 / 0.18 = -0.55555...
        check("BCELoss grad[0]", Math.abs(pred.grad.data[0] - 0.55555f) < 1e-4f);
        check("BCELoss grad[1]", Math.abs(pred.grad.data[1] - (-0.55555f)) < 1e-4f);
    }

    private static void testBCEWithLogitsLoss() {
        System.out.println("Testing BCEWithLogitsLoss...");
        Tensor logits = new Tensor(new float[]{0.0f, 2.0f}, 2);
        logits.requires_grad = true;
        Tensor target = new Tensor(new float[]{0.0f, 1.0f}, 2);
        Tensor loss = nn.F.binary_cross_entropy_with_logits(logits, target);
        // sample 0: sig(0)=0.5. Loss = - (0*log(0.5) + (1-0)*log(1-0.5)) = -log(0.5) = 0.6931
        // sample 1: sig(2)=0.8808. Loss = - (1*log(0.8808) + (1-1)*log(0.1192)) = -log(0.8808) = 0.1269
        // mean = (0.6931 + 0.1269)/2 = 0.4100
        check("BCEWithLogits value", Math.abs(loss.data[0] - 0.4100f) < 1e-3f);

        loss.backward();
        // grad = (sig(logits) - target) / n
        // [0.5-0]/2 = 0.25
        // [0.8808-1]/2 = -0.0596
        check("BCEWithLogits grad[0]", Math.abs(logits.grad.data[0] - 0.25f) < 1e-4f);
        check("BCEWithLogits grad[1]", Math.abs(logits.grad.data[1] - (-0.0596f)) < 1e-4f);
    }

    private static void testKLDivLoss() {
        System.out.println("Testing KLDivLoss...");
        Tensor input = new Tensor(new float[]{-1f, -2f}, 2); // log-probs
        input.requires_grad = true;
        Tensor target = new Tensor(new float[]{0.5f, 0.5f}, 2); // target probs
        Tensor loss = nn.F.kl_div(input, target);
        // Loss = Q * (log(Q) - input)
        // sample 0: 0.5 * (log(0.5) - (-1)) = 0.5 * (-0.6931 + 1) = 0.1534
        // sample 1: 0.5 * (log(0.5) - (-2)) = 0.5 * (-0.6931 + 2) = 0.6534
        // mean = (0.1534 + 0.6534) / 2 = 0.4034
        check("KLDiv value", Math.abs(loss.data[0] - 0.4034f) < 1e-3f);

        loss.backward();
        // grad = -Q / n = [-0.5, -0.5] / 2 = [-0.25, -0.25]
        check("KLDiv grad[0]", Math.abs(input.grad.data[0] - (-0.25f)) < 1e-4f);
    }

    private static void testCosineSimilarity() {
        System.out.println("Testing CosineSimilarity...");
        Tensor x1 = new Tensor(new float[]{1f, 0f, 1f, 1f}, 2, 2);
        x1.requires_grad = true;
        Tensor x2 = new Tensor(new float[]{0f, 1f, 1f, 1f}, 2, 2);
        x2.requires_grad = true;
        Tensor sim = nn.F.cosine_similarity(x1, x2, 1, 1e-8f);
        
        check("Cosine sim[0]", Math.abs(sim.data[0] - 0f) < 1e-6f);
        check("Cosine sim[1]", Math.abs(sim.data[1] - 1f) < 1e-6f);

        Tensor loss = Torch.sumTensor(sim);
        loss.backward();
        // row 0: grad_x1 = [0,1]. (x2/[1*1] - 0*x1/1) = [0,1]
        check("Cosine grad_x1[0]", Math.abs(x1.grad.data[0] - 0f) < 1e-6f);
        check("Cosine grad_x1[1]", Math.abs(x1.grad.data[1] - 1f) < 1e-6f);
        // row 1: grad_x1 = [0,0]
        check("Cosine grad_x1[2]", Math.abs(x1.grad.data[2] - 0f) < 1e-6f);
        check("Cosine grad_x1[3]", Math.abs(x1.grad.data[3] - 0f) < 1e-6f);
    }

    private static void testPairwiseDistance() {
        System.out.println("Testing PairwiseDistance...");
        Tensor x1 = new Tensor(new float[]{0f, 0f, 3f, 0f}, 2, 2);
        x1.requires_grad = true;
        Tensor x2 = new Tensor(new float[]{3f, 4f, 0f, 0f}, 2, 2);
        x2.requires_grad = true;
        Tensor dist = nn.F.pairwise_distance(x1, x2, 2.0f, 1e-6f);

        check("Pairwise dist[0]", Math.abs(dist.data[0] - 5.0f) < 1e-6f);
        check("Pairwise dist[1]", Math.abs(dist.data[1] - 3.0f) < 1e-6f);

        Tensor loss = Torch.sumTensor(dist);
        loss.backward();
        // grad = (x1-x2)/dist
        // row 0: [-3, -4]/5 = [-0.6, -0.8]
        check("Pairwise grad_x1[0]", Math.abs(x1.grad.data[0] - (-0.6f)) < 1e-6f);
        check("Pairwise grad_x1[1]", Math.abs(x1.grad.data[1] - (-0.8f)) < 1e-6f);
        // row 1: [3, 0]/3 = [1, 0]
        check("Pairwise grad_x1[2]", Math.abs(x1.grad.data[2] - 1.0f) < 1e-6f);
    }
}
