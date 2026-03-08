package com.user.nn.containers;

import com.user.nn.core.*;
import com.user.nn.core.Module;

public class ModuleDict extends Module {
    public void put(String name, Module m) {
        addModule(name, m);
    }

    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException("ModuleDict does not implement forward directly");
    }

    @Override
    public NN.Mat forward(NN.Mat x) {
        throw new UnsupportedOperationException("ModuleDict does not implement forward directly");
    }
}
