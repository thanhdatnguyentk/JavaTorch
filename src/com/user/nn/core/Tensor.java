package com.user.nn.core;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import jcuda.*;
import jcuda.runtime.JCuda;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public class Tensor implements AutoCloseable {
    public enum Device { CPU, GPU }
    public Device device = Device.CPU;
    
    public boolean isGPU() {
        return device == Device.GPU;
    }

    public int[] shape;
    public float[] data;
    
    // GPU memory
    public Pointer deviceData = null;
    boolean poolManaged = false; // true if allocated from GpuMemoryPool
    
    // Cleaner for GPU memory safety net (replaces deprecated finalize())
    private static final Cleaner CLEANER = Cleaner.create();
    private Cleaner.Cleanable cleanable;
    private CleanAction cleanAction;
    
    // Static so it holds no reference to the enclosing Tensor (would prevent GC)
    private static class CleanAction implements Runnable {
        Pointer deviceData;
        boolean poolManaged;
        CleanAction(Pointer deviceData, boolean poolManaged) {
            this.deviceData = deviceData;
            this.poolManaged = poolManaged;
        }
        @Override
        public void run() {
            if (deviceData != null && !poolManaged) {
                cudaFree(deviceData);
            }
            deviceData = null;
        }
    }
    
    // Version counter for in-place mutation detection
    private int _version = 0;

    public int version() { return _version; }

    void incrementVersion() { _version++; }

    // Synchronization flags
    private boolean onHost = true;
    private boolean onDevice = false;

    // autograd fields
    public boolean requires_grad = false;
    public Tensor grad = null;
    public GradFn grad_fn = null;

    public Tensor(int... shape) {
        this.shape = shape.clone();
        int size = 1;
        for (int s : shape)
            size *= s;
        this.data = new float[size];
        
        MemoryScope scope = MemoryScope.current();
        if (scope != null) {
            scope.track(this);
        }
    }

    public Tensor(float[] data, int... shape) {
        int n = 1;
        for (int s : shape)
            n *= s;
        if (data.length != n)
            throw new IllegalArgumentException("data length does not match shape");
        this.shape = shape.clone();
        this.data = data.clone();
    }

    public int dim() {
        return shape.length;
    }

    public int numel() {
        int n = 1;
        for (int s : shape)
            n *= s;
        return n;
    }

    public Tensor reshape(int... newShape) {
        this.toCPU();
        int totalNumel = numel();
        int[] ns = newShape.clone();
        int minusOneIdx = -1;
        int product = 1;
        for (int i = 0; i < ns.length; i++) {
            if (ns[i] == -1) {
                if (minusOneIdx != -1) throw new IllegalArgumentException("reshape: multiple -1");
                minusOneIdx = i;
            } else {
                product *= ns[i];
            }
        }
        if (minusOneIdx != -1) {
            if (product == 0 || totalNumel % product != 0) throw new IllegalArgumentException("reshape: incompatible -1");
            ns[minusOneIdx] = totalNumel / product;
        }
        
        int n = 1;
        for (int s : ns) n *= s;
        if (n != totalNumel)
            throw new IllegalArgumentException("reshape: incompatible shape, expected " + totalNumel + " got " + n);
        Tensor result = new Tensor(this.data, ns);
        if (Torch.is_grad_enabled() && this.requires_grad) {
            result.requires_grad = true;
            result.grad_fn = new GradFn(this) {
                public void apply(Tensor gradOutput) {
                    backwardStep(gradOutput.reshape(shape));
                }
            };
        }
        return result;
    }

    public Tensor view(int... newShape) {
        return reshape(newShape);
    }

    public Tensor clone() {
        this.toCPU(); // Ensure data is up-to-date
        return new Tensor(this.data, this.shape);
    }

    /**
     * Return a tensor detached from the current computation graph.
     * The returned tensor will not require gradients and has no grad_fn.
     */
    public Tensor detach() {
        Tensor t = this.clone();
        if (this.device == Device.GPU) t.toGPU();
        t.requires_grad = false;
        t.grad_fn = null;
        return t;
    }

    // Autograd support
    public static abstract class GradFn {
        public final Tensor[] dependencies;
        private final int[] savedVersions;

        public GradFn(Tensor... dependencies) {
            this.dependencies = dependencies;
            this.savedVersions = new int[dependencies.length];
            for (int i = 0; i < dependencies.length; i++) {
                this.savedVersions[i] = dependencies[i].version();
            }
        }

        public void checkVersions() {
            for (int i = 0; i < dependencies.length; i++) {
                if (dependencies[i].version() != savedVersions[i]) {
                    throw new RuntimeException(
                        "one of the variables needed for gradient computation has been "
                        + "modified by an in-place operation (expected version "
                        + savedVersions[i] + ", got " + dependencies[i].version() + ")");
                }
            }
        }

        public abstract void apply(Tensor gradOutput);
    }

    // accumulate gradient (elementwise add)
    public void accumulateGrad(Tensor g) {
        if (this.grad == null) {
            this.grad = g.clone();
            this.grad.toCPU();
        } else {
            this.grad.toCPU();
            g.toCPU();
            if (this.grad.data.length != g.data.length)
                throw new IllegalArgumentException("grad size mismatch");
            for (int i = 0; i < this.grad.data.length; i++)
                this.grad.data[i] += g.data[i];
            this.grad.markDirtyOnCPU();
        }
    }

    /** Zero out accumulated gradient for this tensor. */
    public void zero_grad() {
        this.grad = null;
    }

    // backward entry point (assumes this is a scalar or user provided)
    public void backward() {
        if (!this.requires_grad)
            throw new IllegalStateException("backward() called on tensor that does not require grad");
        
        // Topological sort
        java.util.List<Tensor> topo = new java.util.ArrayList<>();
        java.util.Set<Tensor> visited = new java.util.HashSet<>();
        buildTopo(this, visited, topo);

        Tensor gradInit = new Tensor(this.shape);
        for (int i = 0; i < gradInit.data.length; i++)
            gradInit.data[i] = 1f;
        this.grad = gradInit; // Root gradient is 1.0

        // Iterate in reverse topological order (root -> leaves)
        for (int i = topo.size() - 1; i >= 0; i--) {
            Tensor t = topo.get(i);
            if (t.grad_fn != null && t.grad != null) {
                t.grad_fn.checkVersions();
                t.grad_fn.apply(t.grad);
            }
        }
    }

    /**
     * Backward with a provided initial gradient tensor (for non-scalar roots).
     */
    public void backward(Tensor grad) {
        if (!this.requires_grad)
            throw new IllegalStateException("backward() called on tensor that does not require grad");

        // Topological sort
        java.util.List<Tensor> topo = new java.util.ArrayList<>();
        java.util.Set<Tensor> visited = new java.util.HashSet<>();
        buildTopo(this, visited, topo);

        this.grad = grad.clone();

        // Iterate in reverse topological order (root -> leaves)
        for (int i = topo.size() - 1; i >= 0; i--) {
            Tensor t = topo.get(i);
            if (t.grad_fn != null && t.grad != null) {
                t.grad_fn.checkVersions();
                t.grad_fn.apply(t.grad);
            }
        }
    }

    private void buildTopo(Tensor t, java.util.Set<Tensor> visited, java.util.List<Tensor> topo) {
        if (!visited.contains(t)) {
            visited.add(t);
            if (t.grad_fn != null && t.grad_fn.dependencies != null) {
                for (Tensor dep : t.grad_fn.dependencies) {
                    if (dep != null) {
                        buildTopo(dep, visited, topo);
                    }
                }
            }
            topo.add(t);
        }
    }

    // package-visible for GradFn callbacks
    // Only accumulates grad now, does not trigger recursive backward
    public void backwardStep(Tensor gradOutput) {
        accumulateGrad(gradOutput);
    }

    // Indexing helpers (row-major)
    public float get(int... idx) {
        this.toCPU();
        return data[offset(idx)];
    }

    public void set(float value, int... idx) {
        this.toCPU();
        data[offset(idx)] = value;
        this.markDirtyOnCPU();
        incrementVersion();
    }

    public int offset(int... idx) {
        if (idx.length != shape.length)
            throw new IllegalArgumentException("index rank mismatch");
        int off = 0;
        for (int i = 0; i < shape.length; i++) {
            if (idx[i] < 0 || idx[i] >= shape[i])
                throw new IndexOutOfBoundsException("index out of range");
            off = off * shape[i] + idx[i];
        }
        return off;
    }

    // Elementwise operations returning new Tensor
    public Tensor add(float scalar) {
        this.toCPU();
        Tensor out = new Tensor(shape);
        for (int i = 0; i < data.length; i++)
            out.data[i] = data[i] + scalar;
        return out;
    }

    public Tensor mul(float scalar) {
        this.toCPU();
        Tensor out = new Tensor(shape);
        for (int i = 0; i < data.length; i++)
            out.data[i] = data[i] * scalar;
        return out;
    }

    // In-place ops
    public void add_(float scalar) {
        this.toCPU();
        for (int i = 0; i < data.length; i++)
            data[i] += scalar;
        this.markDirtyOnCPU();
        incrementVersion();
    }

    public void mul_(float scalar) {
        this.toCPU();
        for (int i = 0; i < data.length; i++)
            data[i] *= scalar;
        this.markDirtyOnCPU();
        incrementVersion();
    }

    public void sub_(float scalar) {
        this.toCPU();
        for (int i = 0; i < data.length; i++)
            data[i] -= scalar;
        this.markDirtyOnCPU();
        incrementVersion();
    }

    public void add_(Tensor other) {
        if (this.numel() != other.numel())
            throw new IllegalArgumentException("add_: size mismatch");
        if (this.isGPU() && other.isGPU() && CUDAOps.isAvailable()) {
            CUDAOps.addInPlace(this, other);
        } else {
            this.toCPU();
            other.toCPU();
            for (int i = 0; i < data.length; i++)
                data[i] += other.data[i];
            this.markDirtyOnCPU();
        }
        incrementVersion();
    }

    public void sub_(Tensor other) {
        if (this.numel() != other.numel())
            throw new IllegalArgumentException("sub_: size mismatch");
        if (this.isGPU() && other.isGPU() && CUDAOps.isAvailable()) {
            CUDAOps.subInPlace(this, other);
        } else {
            this.toCPU();
            other.toCPU();
            for (int i = 0; i < data.length; i++)
                data[i] -= other.data[i];
            this.markDirtyOnCPU();
        }
        incrementVersion();
    }

    public void mul_(Tensor other) {
        if (this.numel() != other.numel())
            throw new IllegalArgumentException("mul_: size mismatch");
        if (this.isGPU() && other.isGPU() && CUDAOps.isAvailable()) {
            CUDAOps.mulInPlace(this, other);
        } else {
            this.toCPU();
            other.toCPU();
            for (int i = 0; i < data.length; i++)
                data[i] *= other.data[i];
            this.markDirtyOnCPU();
        }
        incrementVersion();
    }

    // shape utilities
    public int[] shape() {
        return shape.clone();
    }

    public Tensor flatten() {
        return reshape(numel());
    }

    public Tensor squeeze() {
        // remove dimensions of size 1
        int cnt = 0;
        for (int s : shape)
            if (s != 1)
                cnt++;
        if (cnt == shape.length)
            return this; // nothing to squeeze
        int[] ns = new int[cnt];
        int p = 0;
        for (int s : shape)
            if (s != 1)
                ns[p++] = s;
        return reshape(ns);
    }

    public Tensor unsqueeze(int dim) {
        if (dim < 0)
            dim += shape.length + 1;
        if (dim < 0 || dim > shape.length)
            throw new IndexOutOfBoundsException("unsqueeze dim " + dim + " out of range for tensor with " + shape.length + " dimensions");
        int[] ns = new int[shape.length + 1];
        for (int i = 0; i < ns.length; i++)
            ns[i] = (i == dim) ? 1 : 0;
        int pi = 0;
        for (int i = 0; i < ns.length; i++)
            if (ns[i] == 0)
                ns[i] = shape[pi++];
        return reshape(ns);
    }

    public String toString() {
        if (!onHost && onDevice) {
            // Need data for printing
            toCPU();
        }
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=" + Arrays.toString(shape) + ", device=" + device + ", data=");
        int limit = Math.min(data.length, 20);
        sb.append("[");
        String fmt = "%." + Torch.printOptions.precision + "f";
        for (int i = 0; i < limit; i++) {
            if (i > 0)
                sb.append(", ");
            sb.append(String.format(fmt, data[i]));
        }
        if (data.length > limit)
            sb.append(", ... " + data.length + " elements");
        sb.append("])");
        return sb.toString();
    }

    // --- JCuda Sync Methods ---

    public Tensor to(Device targetDevice) {
        if (targetDevice == Device.GPU) return toGPU();
        else return toCPU();
    }

    public Tensor toGPU() {
        if (!CUDAOps.isAvailable()) {
            System.err.println("Warning: CUDA not available; staying on CPU for tensor operations.");
            return this;
        }
        if (device == Device.GPU && onDevice && !onHost)
            return this;
        if (deviceData == null) {
            // Only use Arena Pool for short-lived tensors (inside a MemoryScope).
            // Model parameters (allocated outside any scope) use regular cudaMalloc
            // so they won't be overwritten when the pool resets.
            if (MemoryScope.current() != null) {
                Pointer poolSlice = GpuMemoryPool.allocate(numel());
                if (poolSlice != null) {
                    deviceData = poolSlice;
                    poolManaged = true;
                } else {
                    deviceData = new Pointer();
                    cudaMalloc(deviceData, (long) numel() * Sizeof.FLOAT);
                    poolManaged = false;
                }
            } else {
                deviceData = new Pointer();
                cudaMalloc(deviceData, (long) numel() * Sizeof.FLOAT);
                poolManaged = false;
            }
        }
        cudaMemcpy(deviceData, Pointer.to(data), (long) numel() * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        onDevice = true;
        onHost = false; // Mark host as potentially stale
        device = Device.GPU;
        // Register Cleaner as safety net for GPU memory
        cleanAction = new CleanAction(deviceData, poolManaged);
        cleanable = CLEANER.register(this, cleanAction);
        return this;
    }

    public Tensor toCPU() {
        if (device == Device.CPU && onHost && !onDevice)
            return this;
        if (deviceData == null) {
            onHost = true;
            onDevice = false;
            device = Device.CPU;
            return this;
        }
        cudaMemcpy(Pointer.to(data), deviceData, (long) numel() * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        onHost = true;
        onDevice = false; // Mark device as potentially stale
        device = Device.CPU;
        return this;
    }

    public Pointer getDevicePointer() {
        if (!CUDAOps.isAvailable()) {
            throw new IllegalStateException("CUDA not available: no device pointer present");
        }
        if (!onDevice || device != Device.GPU)
            toGPU();
        return deviceData;
    }

    public void markDirtyOnGPU() {
        onDevice = true;
        onHost = false;
        device = Device.GPU;
    }

    public void markDirtyOnCPU() {
        onHost = true;
        onDevice = false;
        device = Device.CPU;
    }

    @Override
    public void close() {
        if (deviceData != null) {
            // Only cudaFree if NOT managed by the memory pool
            if (!poolManaged) {
                cudaFree(deviceData);
            }
            deviceData = null;
            onDevice = false;
            poolManaged = false;
        }
        // Deregister Cleaner (idempotent) to prevent double-free
        if (cleanAction != null) {
            cleanAction.deviceData = null; // Prevent Cleaner from freeing again
            cleanAction = null;
        }
        if (cleanable != null) {
            cleanable.clean();
            cleanable = null;
        }
        if (grad != null) {
            grad.close();
        }
    }

    public float item() {
        this.toCPU();
        if (numel() != 1) 
            throw new IllegalStateException("item() can only be called on singleton tensors (numel=" + numel() + ")");
        return data[0];
    }
}
