package com.user.nn.core;

public class Parameter {
    public NN.Mat data; // legacy
    public Tensor tensorData; // modern
    public boolean requiresGrad = true;

    public Parameter(NN.Mat data) {
        this.data = data;
        this.tensorData = Torch.fromMat(data);
        this.tensorData.requires_grad = this.requiresGrad;
    }

    public Parameter(Tensor tensor) {
        this.tensorData = tensor;
        this.tensorData.requires_grad = this.requiresGrad;
    }

    public Tensor getTensor() {
        this.tensorData.requires_grad = this.requiresGrad;
        return this.tensorData;
    }

    public Tensor getGrad() {
        return this.tensorData.grad;
    }

    public Parameter toGPU() {
        if (this.tensorData != null)
            this.tensorData.toGPU();
        return this;
    }

    public Parameter toCPU() {
        if (this.tensorData != null)
            this.tensorData.toCPU();
        return this;
    }
}
