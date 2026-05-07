package com.user.nn;

import com.user.nn.core.*;
import com.user.nn.layers.*;
import com.user.nn.containers.*;
import com.user.nn.activations.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestContainers {

    @Test
    void testSequential() {
        Sequential seq = new Sequential();
        Linear l1 = new Linear(4, 3, true);
        ReLU r = new ReLU();
        seq.add(l1);
        seq.add(r);

        // Input
        Tensor in = Torch.ones(2, 4);
        Tensor out = seq.forward(in);
        
        assertEquals(2, out.shape[0], "Sequential produced wrong batch size");
        assertEquals(3, out.shape[1], "Sequential produced wrong output channels");
    }

    @Test
    void testModuleList() {
        ModuleList ml = new ModuleList();
        Linear l1 = new Linear(4, 3, true);
        ml.add(l1);
        assertNotNull(ml.get(0), "ModuleList get should return added module");
        assertEquals(2, ml.parameters().size(), "ModuleList should aggregate parameters (weight + bias)");
    }

    @Test
    void testModuleDict() {
        ModuleDict md = new ModuleDict();
        Linear l1 = new Linear(4, 3, true);
        md.put("layer", l1);
        assertNotNull(md.getModule("layer"), "ModuleDict get should return added module");
        assertEquals(l1, md.getModule("layer"));
    }
}
