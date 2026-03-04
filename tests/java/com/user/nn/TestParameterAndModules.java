package com.user.nn;

public class TestParameterAndModules {
    public static void main(String[] args) {
        nn lib = new nn();
        nn.Linear l = new nn.Linear(lib, 6, 2, true);
        // parameters should include weight and bias
        if (l.getParameter("weight") == null) { System.err.println("weight missing"); System.exit(1); }
        if (l.getParameter("bias") == null) { System.err.println("bias missing"); System.exit(2); }

        // parameters() should return non-empty
        if (l.parameters().size() < 1) { System.err.println("parameters() empty"); System.exit(3); }

        // modules() empty for linear
        if (l.modules().size() != 0) { System.err.println("modules() expected empty"); System.exit(4); }

        System.out.println("TEST PASSED: ParametersAndModules");
    }
}
