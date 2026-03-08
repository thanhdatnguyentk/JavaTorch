package com.user.nn.core;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.io.*;

public abstract class Module {
    protected Map<String, Module> children = new LinkedHashMap<>();
    protected Map<String, Parameter> params = new LinkedHashMap<>();
    protected boolean training = true;

    public void train() {
        this.training = true;
        for (Module m : children.values()) {
            m.train();
        }
    }

    public void eval() {
        this.training = false;
        for (Module m : children.values()) {
            m.eval();
        }
    }

    public void to(Tensor.Device device) {
        if (device == Tensor.Device.GPU) toGPU();
        else toCPU();
    }

    public void toGPU() {
        for (Parameter p : params.values()) {
            p.toGPU();
        }
        for (Module m : children.values()) {
            m.toGPU();
        }
    }

    public void toCPU() {
        for (Parameter p : params.values()) {
            p.toCPU();
        }
        for (Module m : children.values()) {
            m.toCPU();
        }
    }

    public boolean is_training() {
        return training;
    }

    public void addModule(String name, Module m) {
        children.put(name, m);
    }

    public void addParameter(String name, Parameter p) {
        params.put(name, p);
    }

    public Module getModule(String name) {
        return children.get(name);
    }

    public Parameter getParameter(String name) {
        return params.get(name);
    }

    public List<Parameter> parameters() {
        List<Parameter> out = new ArrayList<>();
        out.addAll(params.values());
        for (Module m : children.values()) {
            out.addAll(m.parameters());
        }
        return out;
    }

    public void zero_grad() {
        for (Parameter p : parameters()) {
            p.getTensor().grad = null;
        }
    }

    public void save(String path) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(path))) {
            List<Parameter> allParams = parameters();
            dos.writeInt(allParams.size());
            for (Parameter p : allParams) {
                Tensor t = p.getTensor();
                boolean wasGPU = t.isGPU();
                if (wasGPU) t.toCPU();
                dos.writeInt(t.shape.length);
                for (int s : t.shape) dos.writeInt(s);
                dos.writeInt(t.data.length);
                for (float v : t.data) dos.writeFloat(v);
                if (wasGPU) t.toGPU();
            }
        }
    }

    public void load(String path) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(path))) {
            List<Parameter> allParams = parameters();
            int count = dis.readInt();
            if (count != allParams.size()) {
                throw new IOException("Parameter count mismatch: file has " + count + ", model has " + allParams.size());
            }
            for (Parameter p : allParams) {
                int dims = dis.readInt();
                int[] shape = new int[dims];
                for (int i = 0; i < dims; i++) shape[i] = dis.readInt();
                int dataLen = dis.readInt();
                float[] data = new float[dataLen];
                for (int i = 0; i < dataLen; i++) data[i] = dis.readFloat();
                
                Tensor t = p.getTensor();
                if (t.data.length != dataLen) {
                    throw new IOException("Parameter data length mismatch");
                }
                System.arraycopy(data, 0, t.data, 0, dataLen);
                t.markDirtyOnCPU();
                if (t.isGPU()) t.toGPU();
            }
        }
    }

    public long countParameters() {
        long total = 0;
        for (Parameter p : parameters()) {
            total += p.getTensor().numel();
        }
        return total;
    }

    public List<Module> modules() {
        List<Module> out = new ArrayList<>();
        out.addAll(children.values());
        return out;
    }

    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException(
                this.getClass().getSimpleName() + " does not implement forward(Tensor) directly");
    }

    public NN.Mat forward(NN.Mat x) {
        Tensor t = Torch.fromMat(x);
        Tensor out = forward(t);
        NN.Mat m;
        if (out.dim() == 2)
            m = NN.mat_alloc(out.shape[0], out.shape[1]);
        else
            m = NN.mat_alloc(out.shape[0], out.numel() / out.shape[0]);
        System.arraycopy(out.data, 0, m.es, 0, m.es.length);
        return m;
    }

    public Tensor apply(Tensor x) {
        return forward(x);
    }

    public NN.Mat apply(NN.Mat x) {
        return forward(x);
    }
}
