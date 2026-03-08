package com.user.nn.core;

import java.util.ArrayList;
import java.util.List;

/**
 * A context manager (try-with-resources) for keeping track of Tensors
 * and releasing their GPU memory collectively, bypassing slow Java Garbage Collection.
 */
public class MemoryScope implements AutoCloseable {
    private static final ThreadLocal<MemoryScope> currentScope = new ThreadLocal<>();
    private final MemoryScope parent;
    private final List<Tensor> trackedTensors = new ArrayList<>();

    public MemoryScope() {
        this.parent = currentScope.get();
        currentScope.set(this);
    }

    public static MemoryScope current() {
        return currentScope.get();
    }

    /**
     * Track a tensor in this scope.
     * When the scope is closed, all tracked tensors are freed from GPU.
     */
    public void track(Tensor t) {
        trackedTensors.add(t);
    }

    /**
     * Detach a tensor from this scope (e.g., if returning it from a function).
     */
    public void detach(Tensor t) {
        trackedTensors.remove(t);
    }

    @Override
    public void close() {
        RuntimeException firstException = null;
        for (int i = trackedTensors.size() - 1; i >= 0; i--) {
            try {
                trackedTensors.get(i).close();
            } catch (RuntimeException e) {
                if (firstException == null) firstException = e;
                else firstException.addSuppressed(e);
            }
        }
        trackedTensors.clear();
        
        // If this is a top-level scope, reset the memory pool
        if (parent == null && GpuMemoryPool.isInitialized()) {
            GpuMemoryPool.reset();
        }
        
        currentScope.set(parent);
        if (firstException != null) throw firstException;
    }
}
