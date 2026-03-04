package com.user.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Torch {
    public static String defaultDtype = "float32";
    public static PrintOptions printOptions = new PrintOptions();
    private static Random globalR = new Random(0);

    public static class PrintOptions { public int precision = 6; public int threshold = 1000; }

    // dtype helpers
    public static boolean is_tensor(Object o) { return o instanceof Tensor; }
    public static boolean is_floating_point(Tensor t) { return t != null; }
    public static void set_default_dtype(String d) { defaultDtype = d; }
    public static String get_default_dtype() { return defaultDtype; }
    public static void set_printoptions(int precision, int threshold) { printOptions.precision = precision; printOptions.threshold = threshold; }

    public static void manual_seed(long s) { globalR = new Random(s); }

    // Creation
    public static Tensor tensor(float[] data, int... shape) { return new Tensor(data, shape); }
    public static Tensor zeros(int... shape) { Tensor t = new Tensor(shape); for (int i=0;i<t.data.length;i++) t.data[i]=0f; return t; }
    public static Tensor ones(int... shape) { Tensor t = new Tensor(shape); for (int i=0;i<t.data.length;i++) t.data[i]=1f; return t; }
    public static Tensor full(int[] shape, float value) { Tensor t = new Tensor(shape); for (int i=0;i<t.data.length;i++) t.data[i]=value; return t; }
    public static Tensor arange(int start, int end) { int n = end - start; float[] d = new float[n]; for (int i=0;i<n;i++) d[i]=start+i; return new Tensor(d, n); }
    public static Tensor eye(int n) { Tensor t = zeros(n,n); for (int i=0;i<n;i++) t.data[i*n+i]=1f; return t; }

    public static Tensor rand(int[] shape) { Tensor t = new Tensor(shape); for (int i=0;i<t.data.length;i++) t.data[i]=globalR.nextFloat(); return t; }
    public static Tensor randn(int[] shape) { Tensor t = new Tensor(shape); for (int i=0;i<t.data.length;i++) t.data[i]=(float)nextGaussian(globalR); return t; }

    private static double nextGaussian(Random r) { // Box-Muller
        double u = r.nextDouble(); double v = r.nextDouble(); return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v);
    }

    // Basic math (elementwise) - assumes same shape
    public static Tensor add(Tensor a, Tensor b) { return binaryOp(a,b,(x,y)->x+y); }
    public static Tensor sub(Tensor a, Tensor b) { return binaryOp(a,b,(x,y)->x-y); }
    public static Tensor mul(Tensor a, Tensor b) { return binaryOp(a,b,(x,y)->x*y); }
    public static Tensor div(Tensor a, Tensor b) { return binaryOp(a,b,(x,y)->x/y); }

    private static Tensor binaryOp(Tensor a, Tensor b, FloatBinaryOp op) {
        if (a.numel() != b.numel()) throw new IllegalArgumentException("broadcast not implemented");
        Tensor out = new Tensor(a.shape);
        for (int i=0;i<a.data.length;i++) out.data[i] = op.apply(a.data[i], b.data[i]);
        return out;
    }

    private interface FloatBinaryOp { float apply(float x, float y); }

    // reductions
    public static float sum(Tensor a) { float s=0f; for (float v: a.data) s+=v; return s; }
    public static float mean(Tensor a) { return sum(a)/a.numel(); }

    // reshape / transpose utilities (simple)
    public static Tensor reshape(Tensor a, int... newShape) { return a.reshape(newShape); }
    public static Tensor view(Tensor a, int... newShape) { return reshape(a,newShape); }

    public static Tensor transpose(Tensor a, int dim0, int dim1) {
        if (a.shape.length != 2) throw new IllegalArgumentException("transpose only supports 2D currently");
        int r = a.shape[0], c = a.shape[1]; Tensor out = new Tensor(c,r);
        for (int i=0;i<r;i++) for (int j=0;j<c;j++) out.data[j*r + i] = a.data[i*c + j];
        return out;
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        if (a.shape.length != 2 || b.shape.length != 2) throw new IllegalArgumentException("matmul supports 2D tensors");
        int m = a.shape[0], k = a.shape[1], n = b.shape[1];
        if (k != b.shape[0]) throw new IllegalArgumentException("matmul shape mismatch");
        Tensor out = new Tensor(m, n);
        for (int i=0;i<m;i++) for (int j=0;j<n;j++){
            float s=0f; for (int kk=0;kk<k;kk++) s += a.data[i*k + kk] * b.data[kk * n + j]; out.data[i*n + j] = s;
        }
        return out;
    }

    // convert between nn.Mat (2D) and Tensor
    public static Tensor fromMat(nn.Mat m) {
        return new Tensor(m.es.clone(), m.rows, m.cols);
    }
    public static nn.Mat toMat(Tensor t) {
        if (t.shape.length != 2) throw new IllegalArgumentException("toMat requires 2D tensor");
        nn outer = new nn();
        nn.Mat m = outer.mat_alloc(t.shape[0], t.shape[1]);
        System.arraycopy(t.data, 0, m.es, 0, t.data.length);
        return m;
    }

    // concatenation along dim 0 (simple)
    public static Tensor cat(List<Tensor> tensors, int dim) {
        if (tensors.size()==0) throw new IllegalArgumentException("no tensors to cat");
        if (dim != 0) throw new UnsupportedOperationException("cat currently supports dim=0 only");
        int rows = 0; int cols = tensors.get(0).shape[1];
        for (Tensor t: tensors) { if (t.shape.length!=2 || t.shape[1]!=cols) throw new IllegalArgumentException("incompatible shapes"); rows += t.shape[0]; }
        Tensor out = new Tensor(rows, cols); int pos=0; for (Tensor t: tensors){ System.arraycopy(t.data,0,out.data,pos*t.shape[1], t.data.length); pos += t.shape[0]; }
        return out;
    }

    // argmax along axis 1 (rows)
    public static int[] argmax(Tensor a, int axis) {
        if (axis != 1 || a.shape.length!=2) throw new UnsupportedOperationException("only axis=1 for 2D implemented");
        int rows = a.shape[0], cols = a.shape[1]; int[] out = new int[rows];
        for (int i=0;i<rows;i++){ int best=0; float bv=a.data[i*cols]; for (int j=1;j<cols;j++){ float v=a.data[i*cols + j]; if (v>bv){bv=v; best=j;} } out[i]=best; }
        return out;
    }

}
