package com.user.nn;

import com.user.nn.core.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestWeightInit {

    @BeforeEach
    void setup() {
        Torch.manual_seed(42);
    }

    private float mean(float[] d) {
        double s = 0; for (float v : d) s += v; return (float)(s / d.length);
    }
    
    private float variance(float[] d, float mean) {
        double s = 0; for (float v : d) s += (v - mean) * (v - mean); return (float)(s / d.length);
    }

    @Test
    void testUniform() {
        Tensor t = new Tensor(1000);
        Torch.nn.init.uniform_(t, -1f, 1f);
        float m = mean(t.data);
        assertTrue(Math.abs(m) < 0.1f, "uniform_ mean near 0, got " + m);
        assertTrue(t.data[0] >= -1f && t.data[0] <= 1f, "Value out of range");
    }

    @Test
    void testNormal() {
        Tensor t = new Tensor(10000);
        Torch.nn.init.normal_(t, 0f, 1f);
        float m = mean(t.data);
        float v = variance(t.data, m);
        assertTrue(Math.abs(m) < 0.05f, "normal_ mean near 0, got " + m);
        assertTrue(Math.abs(v - 1.0f) < 0.1f, "normal_ variance near 1, got " + v);
    }

    @Test
    void testBasicInits() {
        Tensor t = new Tensor(100);
        Torch.nn.init.zeros_(t);
        assertEquals(0f, t.data[0]);
        assertEquals(0f, t.data[99]);

        Torch.nn.init.ones_(t);
        assertEquals(1f, t.data[0]);
        assertEquals(1f, t.data[99]);

        Torch.nn.init.constant_(t, 3.14f);
        assertEquals(3.14f, t.data[0], 1e-6f);
    }

    @Test
    void testXavier() {
        Tensor t = new Tensor(256, 512);
        Torch.nn.init.xavier_uniform_(t);
        float m = mean(t.data);
        float v = variance(t.data, m);
        float expectedVar = 2.0f / (512 + 256);
        assertTrue(Math.abs(m) < 0.01f, "Xavier mean near 0");
        assertTrue(Math.abs(v - expectedVar) < 0.001f, "Xavier uniform variance mismatch");

        Torch.nn.init.xavier_normal_(t);
        m = mean(t.data);
        v = variance(t.data, m);
        assertTrue(Math.abs(m) < 0.01f, "Xavier normal mean near 0");
        assertTrue(Math.abs(v - expectedVar) < 0.001f, "Xavier normal variance mismatch");
    }

    @Test
    void testKaiming() {
        Tensor t = new Tensor(256, 512);
        Torch.nn.init.kaiming_uniform_(t);
        float m = mean(t.data);
        float v = variance(t.data, m);
        float expectedVar = 2.0f / 512;
        assertTrue(Math.abs(m) < 0.01f, "Kaiming mean near 0");
        assertTrue(Math.abs(v - expectedVar) < 0.001f, "Kaiming uniform variance mismatch");

        Torch.nn.init.kaiming_normal_(t);
        m = mean(t.data);
        v = variance(t.data, m);
        assertTrue(Math.abs(m) < 0.01f, "Kaiming normal mean near 0");
        assertTrue(Math.abs(v - expectedVar) < 0.001f, "Kaiming normal variance mismatch");
    }

    @Test
    void testFanCalc() {
        int[] fan = Torch.nn.init.calculateFanInOut(new Tensor(64, 32, 3, 3));
        assertEquals(32 * 3 * 3, fan[0], "fan_in for 4D");
        assertEquals(64 * 3 * 3, fan[1], "fan_out for 4D");
    }

    @Test
    void testGains() {
        assertEquals(Math.sqrt(2.0), Torch.nn.init.calculateGain("relu"), 1e-5f);
        assertEquals(5.0f/3.0f, Torch.nn.init.calculateGain("tanh"), 1e-5f);
        assertEquals(1.0f, Torch.nn.init.calculateGain("linear"), 1e-5f);
    }
}
