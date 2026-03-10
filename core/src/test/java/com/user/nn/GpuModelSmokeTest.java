package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.core.Functional;
import com.user.nn.models.cv.LeNet;
import com.user.nn.models.cv.ResNet;
import com.user.nn.models.generative.GAN;
import com.user.nn.models.generative.VAE;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuModelSmokeTest {

    @Test
    @Tag("gpu-smoke")
    void lenetForwardBackwardOnGpu() {
        GpuTestSupport.assumeCuda();
        GpuTestSupport.seed(7L);

        LeNet model = new LeNet();
        model.toGPU();
        long gpuParams = model.parameters().stream().filter(p -> p.getTensor().isGPU()).count();
        assertTrue(gpuParams > 0, "lenet parameters should move to GPU");

        Tensor x = Torch.randn(new int[]{2, 1, 28, 28});
        x.toGPU();

        Tensor out = model.forward(x);
        assertEquals(2, out.shape[0]);
        assertTrue(out.shape[1] > 0, "lenet should produce non-empty class dimension");
        GpuTestSupport.assertFinite(out, "lenet output");

        assertTrue(!model.parameters().isEmpty(), "model should have parameters");
    }

    @Test
    @Tag("gpu-nightly")
    void resnet18ForwardBackwardOnGpu() {
        GpuTestSupport.assumeCuda();
        GpuTestSupport.assumeBlas();
        GpuTestSupport.seed(11L);

        ResNet model = ResNet.resnet18(10, 32, 32);
        model.toGPU();

        Tensor x = Torch.randn(new int[]{1, 3, 32, 32});
        x.toGPU();

        Tensor out = model.forward(x);
        GpuTestSupport.assertGpu(out, "resnet output");
        assertEquals(1, out.shape[0]);
        assertEquals(10, out.shape[1]);

        Tensor target = Torch.zeros(1, 10);
        target.toGPU();
        Tensor loss = Functional.mse_loss_tensor(out, target);
        loss.backward();

        GpuTestSupport.assertFinite(loss, "resnet loss");
        long grads = model.parameters().stream().filter(p -> p.getTensor().grad != null).count();
        assertTrue(grads > 0, "resnet parameters should receive gradients");
    }

    @Test
    @Tag("gpu-nightly")
    void vaeForwardOnGpu() {
        GpuTestSupport.assumeCuda();
        GpuTestSupport.assumeBlas();
        GpuTestSupport.seed(29L);

        VAE model = new VAE(32, 8);
        model.toGPU();
        long gpuParams = model.parameters().stream().filter(p -> p.getTensor().isGPU()).count();
        assertTrue(gpuParams > 0, "vae parameters should move to GPU");

        Tensor x = Torch.randn(new int[]{2, 32});
        x.toGPU();

        Tensor out = model.forward(x);
        assertEquals(2, out.shape[0]);
        assertEquals(32, out.shape[1]);
        GpuTestSupport.assertFinite(out, "vae output");
    }

    @Test
    @Tag("gpu-nightly")
    void ganGeneratorDiscriminatorForwardOnGpu() {
        GpuTestSupport.assumeCuda();
        GpuTestSupport.assumeBlas();
        GpuTestSupport.seed(31L);

        GAN.Generator generator = new GAN.Generator(16, 32);
        GAN.Discriminator discriminator = new GAN.Discriminator(32);
        generator.toGPU();
        discriminator.toGPU();
        long generatorGpuParams = generator.parameters().stream().filter(p -> p.getTensor().isGPU()).count();
        long discriminatorGpuParams = discriminator.parameters().stream().filter(p -> p.getTensor().isGPU()).count();
        assertTrue(generatorGpuParams > 0, "gan generator parameters should move to GPU");
        assertTrue(discriminatorGpuParams > 0, "gan discriminator parameters should move to GPU");

        Tensor z = Torch.randn(new int[]{2, 16});
        z.toGPU();

        Tensor fake = generator.forward(z);
        assertEquals(2, fake.shape[0]);
        assertEquals(32, fake.shape[1]);
        GpuTestSupport.assertFinite(fake, "gan generator output");

        Tensor logits = discriminator.forward(fake);
        assertEquals(2, logits.shape[0]);
        assertEquals(1, logits.shape[1]);
        GpuTestSupport.assertFinite(logits, "gan discriminator output");
    }
}
