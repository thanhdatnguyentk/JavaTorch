package com.user.nn.containers;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import java.util.ArrayList;
import java.util.List;

public class ModuleList extends Module {
    private final List<Module> list = new ArrayList<>();

    public void add(Module m) {
        String name = "" + list.size();
        list.add(m);
        addModule(name, m);
    }

    public Module get(int idx) {
        return list.get(idx);
    }

    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException("ModuleList does not implement forward directly");
    }

    @Override
    public NN.Mat forward(NN.Mat x) {
        throw new UnsupportedOperationException("ModuleList does not implement forward directly");
    }
}
