package com.user.nn.core;

import java.util.List;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class NN {
    public static class Mat {
        public int rows;
        public int cols;
        public float[] es;
    }

    public static final float MAT_AT(Mat m, int i, int j) {
        return m.es[i * m.cols + j];
    }

    public static Mat mat_alloc(int rows, int cols) {
        Mat m = new Mat();
        m.rows = rows;
        m.cols = cols;
        m.es = new float[rows * cols];
        return m;
    }

    public static void mat_dot(Mat dst, Mat a, Mat b) {
        // Use Tensor-based matmul for correctness and reuse
        if (a.cols != b.rows)
            throw new IllegalArgumentException("Incompatible matrix dimensions for dot: a.cols must equal b.rows");
        if (dst.rows != a.rows || dst.cols != b.cols)
            throw new IllegalArgumentException("Destination matrix has wrong dimensions");
        Tensor ta = Torch.fromMat(a);
        Tensor tb = Torch.fromMat(b);
        Tensor tc = Torch.matmul(ta, tb);
        // copy back
        if (tc.shape.length != 2 || tc.shape[0] != dst.rows || tc.shape[1] != dst.cols)
            throw new IllegalStateException("matmul result shape mismatch");
        System.arraycopy(tc.data, 0, dst.es, 0, dst.es.length);
    }

    public static void mat_sum(Mat dst, Mat a) {
        if (dst.rows != a.rows || dst.cols != a.cols)
            throw new IllegalArgumentException("Matrices must have the same dimensions for addition");
        Tensor ta = Torch.fromMat(dst);
        Tensor tb = Torch.fromMat(a);
        Tensor tr = Torch.add(ta, tb);
        System.arraycopy(tr.data, 0, dst.es, 0, dst.es.length);
    }

    public static void mat_sub(Mat dst, Mat a) {
        if (dst.rows != a.rows || dst.cols != a.cols)
            throw new IllegalArgumentException("Matrices must have the same dimensions for subtraction");
        Tensor ta = Torch.fromMat(dst);
        Tensor tb = Torch.fromMat(a);
        Tensor tr = Torch.sub(ta, tb);
        System.arraycopy(tr.data, 0, dst.es, 0, dst.es.length);
    }

    public static void mat_print(Mat m) {
        Tensor t = Torch.fromMat(m);
        System.out.println(t.toString());
    }

    public static void mat_rand(Mat m, float min, float max) {
        Tensor t = Torch.rand(new int[] { m.rows, m.cols });
        // scale to [min,max)
        float range = max - min;
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = min + t.data[i] * range;
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    public static void mat_fill(Mat m, float value) {
        Tensor t = Torch.full(new int[] { m.rows, m.cols }, value);
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // Elementwise apply (in-place) using a lambda-like helper interface
    public interface ElemOp {
        float apply(float x);
    }

    public static void mat_apply_inplace(Mat m, ElemOp op) {
        // apply elementwise via Tensor
        Tensor t = Torch.fromMat(m);
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = op.apply(t.data[i]);
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // Deterministic random fill
    public static void mat_rand_seed(Mat m, long seed, float min, float max) {
        // deterministic rand via Torch.randn seeded by seed
        Torch.manual_seed(seed);
        Tensor t = Torch.rand(new int[] { m.rows, m.cols });
        float range = max - min;
        for (int i = 0; i < t.data.length; i++)
            t.data[i] = min + t.data[i] * range;
        System.arraycopy(t.data, 0, m.es, 0, m.es.length);
    }

    // CSV read/write utilities (rows lines, comma-separated)
    public static void writeMatCSV(Mat m, String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                sb.append(m.es[i * m.cols + j]);
                if (j + 1 < m.cols)
                    sb.append(',');
            }
            sb.append('\n');
        }
        Files.createDirectories(Paths.get(path).getParent());
        try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(path))) {
            bw.write(sb.toString());
        }
    }

    public static Mat readMatCSV(String path) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(path));
        if (lines.size() == 0)
            return null;
        int rows = lines.size();
        String[] first = lines.get(0).split(",");
        int cols = first.length;
        Mat m = mat_alloc(rows, cols);
        for (int i = 0; i < rows; i++) {
            String[] parts = lines.get(i).trim().split(",");
            for (int j = 0; j < cols; j++) {
                m.es[i * cols + j] = Float.parseFloat(parts[j]);
            }
        }
        return m;
    }

}
