package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestBatch2 {

    @Test
    void testL1Loss() {
        Tensor pred = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, 3);
        pred.requires_grad = true;
        Tensor target = new Tensor(new float[]{1.5f, 1.5f, 3.5f}, 3);
        Tensor loss = Functional.l1_loss(pred, target);
        
        assertEquals(0.5f, loss.data[0], 1e-6f, "L1Loss value mismatch");

        loss.backward();
        assertEquals(-1f/3f, pred.grad.data[0], 1e-6f);
        assertEquals(1f/3f, pred.grad.data[1], 1e-6f);
    }

    @Test
    void testBCELoss() {
        Tensor pred = new Tensor(new float[]{0.1f, 0.9f}, 2);
        pred.requires_grad = true;
        Tensor target = new Tensor(new float[]{0.0f, 1.0f}, 2);
        Tensor loss = Functional.binary_cross_entropy(pred, target);
        
        assertEquals(0.10536f, loss.data[0], 1e-4f, "BCELoss value mismatch");

        loss.backward();
        assertEquals(0.55555f, pred.grad.data[0], 1e-4f);
        assertEquals(-0.55555f, pred.grad.data[1], 1e-4f);
    }

    @Test
    void testBCEWithLogitsLoss() {
        Tensor logits = new Tensor(new float[]{0.0f, 2.0f}, 2);
        logits.requires_grad = true;
        Tensor target = new Tensor(new float[]{0.0f, 1.0f}, 2);
        Tensor loss = Functional.binary_cross_entropy_with_logits(logits, target);
        
        assertEquals(0.4100f, loss.data[0], 1e-3f, "BCEWithLogits value mismatch");

        loss.backward();
        assertEquals(0.25f, logits.grad.data[0], 1e-4f);
        assertEquals(-0.0596f, logits.grad.data[1], 1e-4f);
    }

    @Test
    void testKLDivLoss() {
        Tensor input = new Tensor(new float[]{-1f, -2f}, 2); // log-probs
        input.requires_grad = true;
        Tensor target = new Tensor(new float[]{0.5f, 0.5f}, 2); // target probs
        Tensor loss = Functional.kl_div(input, target);
        
        assertEquals(0.4034f, loss.data[0], 1e-3f, "KLDiv value mismatch");

        loss.backward();
        assertEquals(-0.25f, input.grad.data[0], 1e-4f);
    }

    @Test
    void testCosineSimilarity() {
        Tensor x1 = new Tensor(new float[]{1f, 0f, 1f, 1f}, 2, 2);
        x1.requires_grad = true;
        Tensor x2 = new Tensor(new float[]{0f, 1f, 1f, 1f}, 2, 2);
        x2.requires_grad = true;
        Tensor sim = Functional.cosine_similarity(x1, x2, 1, 1e-8f);
        
        assertEquals(0f, sim.data[0], 1e-6f);
        assertEquals(1f, sim.data[1], 1e-6f);

        Tensor loss = Torch.sumTensor(sim);
        loss.backward();
        assertEquals(0f, x1.grad.data[0], 1e-6f);
        assertEquals(1f, x1.grad.data[1], 1e-6f);
        assertEquals(0f, x1.grad.data[2], 1e-6f);
        assertEquals(0f, x1.grad.data[3], 1e-6f);
    }

    @Test
    void testPairwiseDistance() {
        Tensor x1 = new Tensor(new float[]{0f, 0f, 3f, 0f}, 2, 2);
        x1.requires_grad = true;
        Tensor x2 = new Tensor(new float[]{3f, 4f, 0f, 0f}, 2, 2);
        x2.requires_grad = true;
        Tensor dist = Functional.pairwise_distance(x1, x2, 2.0f, 1e-6f);

        assertEquals(5.0f, dist.data[0], 1e-6f);
        assertEquals(3.0f, dist.data[1], 1e-6f);

        Tensor loss = Torch.sumTensor(dist);
        loss.backward();
        assertEquals(-0.6f, x1.grad.data[0], 1e-6f);
        assertEquals(-0.8f, x1.grad.data[1], 1e-6f);
        assertEquals(1.0f, x1.grad.data[2], 1e-6f);
    }
}
