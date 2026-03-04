package com.user.nn;

import java.util.Arrays;

public class Tensor {
    public int[] shape;
    public float[] data;

    public Tensor(int... shape) {
        this.shape = shape.clone();
        int n = 1; for (int s: shape) n *= s;
        this.data = new float[n];
    }

    public Tensor(float[] data, int... shape) {
        int n = 1; for (int s: shape) n *= s;
        if (data.length != n) throw new IllegalArgumentException("data length does not match shape");
        this.shape = shape.clone();
        this.data = data.clone();
    }

    public int dim() { return shape.length; }

    public int numel() { int n=1; for (int s:shape) n*=s; return n; }

    public Tensor reshape(int... newShape) {
        int n = 1; for (int s: newShape) n*=s;
        if (n != numel()) throw new IllegalArgumentException("reshape: incompatible shape");
        return new Tensor(this.data, newShape);
    }

    public Tensor view(int... newShape) { return reshape(newShape); }

    public Tensor clone() { return new Tensor(this.data, this.shape); }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape="+Arrays.toString(shape)+", data=");
        int limit = Math.min(data.length, 20);
        sb.append("[");
        String fmt = "%." + Torch.printOptions.precision + "f";
        for (int i=0;i<limit;i++){
            if (i>0) sb.append(", ");
            sb.append(String.format(fmt, data[i]));
        }
        if (data.length>limit) sb.append(", ... "+data.length+" elements");
        sb.append("])");
        return sb.toString();
    }
}
