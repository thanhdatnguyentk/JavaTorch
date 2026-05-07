package com.user.nn;

import com.user.nn.layers.Linear;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestParameterAndModules {

    @Test
    void testLinearParametersAndModules() {
        Linear l = new Linear(6, 2, true);
        
        // parameters should include weight and bias
        assertNotNull(l.getParameter("weight"), "weight missing");
        assertNotNull(l.getParameter("bias"), "bias missing");

        // parameters() should return non-empty
        assertTrue(l.parameters().size() >= 2, "parameters() should have at least weight and bias");

        // modules() is empty for basic Linear layer (it doesn't have sub-modules)
        assertEquals(0, l.modules().size(), "modules() expected empty for Linear");
    }
}
