package com.user.nn;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.models.cv.YOLO;
import com.user.nn.models.cv.ViT;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GpuExamplesSmokeTest {

    @Test
    @Tag("gpu-manual")
    void yoloExampleLikeForwardRunsOnGpu() {
        GpuTestSupport.assumeCuda();
        GpuTestSupport.seed(19L);

        // Keep this tiny to avoid OOM in constrained CI JVM heaps.
        YOLO model = new YOLO(2, 32, 32, 3, 1);
        model.toGPU();

        Tensor x = Torch.randn(new int[]{1, 3, 32, 32});
        x.toGPU();

        Tensor out = model.forward(x);
        GpuTestSupport.assertGpu(out, "yolo output");
        assertEquals(1, out.shape[0]);
        GpuTestSupport.assertFinite(out, "yolo output");
    }

    @Test
    @Tag("gpu-nightly")
    void vitExampleLikeForwardRunsOnGpu() {
        GpuTestSupport.assumeCuda();
        GpuTestSupport.assumeBlas();
        GpuTestSupport.seed(23L);

        ViT model = new ViT(32, 4, 3, 10, 64, 1, 4, 128, 0.0f);
        model.toGPU();

        Tensor x = Torch.randn(new int[]{1, 3, 32, 32});
        x.toGPU();

        Tensor out = model.forward(x);
        GpuTestSupport.assertGpu(out, "vit output");
        assertEquals(1, out.shape[0]);
        assertEquals(10, out.shape[1]);
        GpuTestSupport.assertFinite(out, "vit output");
    }
}
