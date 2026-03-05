package com.user.nn;

import java.util.*;
import java.util.concurrent.*;

/**
 * Data loading utilities including Dataset interface and multithreaded
 * DataLoader.
 */
public class data {

    /**
     * Interface for representing a dataset.
     * Each item should return an array of Tensors (e.g. [input_tensor,
     * label_tensor]).
     */
    public interface Dataset {
        /** Returns the total number of samples */
        int len();

        /** Fetches the sample at the given index */
        Tensor[] get(int index);
    }

    /**
     * A multithreaded data loader.
     * Yields batches of data, collated by stacking individual samples.
     */
    public static class DataLoader implements Iterable<Tensor[]>, Iterator<Tensor[]> {
        private final Dataset dataset;
        private final int batchSize;
        private final boolean shuffle;
        private final int numWorkers;

        private Integer[] indices;
        private int currentIdx;
        private ExecutorService executor;

        // Prefetch queue
        private BlockingQueue<Future<Tensor[]>> batchQueue;
        private final int prefetchFactor = 2; // how many batches ahead to queue

        public DataLoader(Dataset dataset, int batchSize, boolean shuffle, int numWorkers) {
            this.dataset = dataset;
            this.batchSize = batchSize;
            this.shuffle = shuffle;
            this.numWorkers = Math.max(1, numWorkers);
            this.indices = new Integer[dataset.len()];
            reset();
        }

        private void reset() {
            int len = dataset.len();
            for (int i = 0; i < len; i++) {
                indices[i] = i;
            }
            if (shuffle) {
                // Determine a fixed or system seed for shuffling
                Collections.shuffle(Arrays.asList(indices), new Random());
            }
            this.currentIdx = 0;

            if (executor != null) {
                executor.shutdownNow();
            }

            // Fixed thread pool for workers
            executor = Executors.newFixedThreadPool(numWorkers, new ThreadFactory() {
                int count = 0;

                public Thread newThread(Runnable r) {
                    Thread t = new Thread(r, "DataLoader-worker-" + (++count));
                    t.setDaemon(true); // Don't block JVM exit
                    return t;
                }
            });

            // Queue capable of holding prefetch batches
            batchQueue = new ArrayBlockingQueue<>(numWorkers * prefetchFactor);

            // Start initial prefetching
            prefetch();
        }

        private void prefetch() {
            while (currentIdx < indices.length && batchQueue.remainingCapacity() > 0) {
                final int start = currentIdx;
                final int end = Math.min(start + batchSize, indices.length);
                currentIdx = end;

                // Submit batch loading task
                Callable<Tensor[]> batchTask = () -> {
                    int size = end - start;
                    List<Tensor[]> samples = new ArrayList<>(size);
                    for (int i = start; i < end; i++) {
                        int idx = indices[i];
                        samples.add(dataset.get(idx));
                    }
                    return collateFn(samples);
                };

                batchQueue.offer(executor.submit(batchTask));
            }
        }

        /**
         * Default collate function.
         * Assumes all samples have the same array length, and tensors of matching
         * shapes.
         * Stacks each position along dimension 0.
         */
        protected Tensor[] collateFn(List<Tensor[]> batchSamples) {
            if (batchSamples.isEmpty())
                return new Tensor[0];
            int numTensors = batchSamples.get(0).length;
            Tensor[] batched = new Tensor[numTensors];

            for (int t = 0; t < numTensors; t++) {
                List<Tensor> tensorList = new ArrayList<>(batchSamples.size());
                for (Tensor[] sample : batchSamples) {
                    tensorList.add(sample[t]);
                }
                batched[t] = Torch.stack(tensorList, 0);
            }
            return batched;
        }

        @Override
        public Iterator<Tensor[]> iterator() {
            reset();
            return this;
        }

        @Override
        public boolean hasNext() {
            return !batchQueue.isEmpty();
        }

        @Override
        public Tensor[] next() {
            if (!hasNext())
                throw new NoSuchElementException();
            try {
                Future<Tensor[]> futureBatch = batchQueue.poll();
                Tensor[] batch = futureBatch.get(); // blocks if computation is not done yet
                // Top up the prefetch queue
                prefetch();
                return batch;
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Error during data loading", e);
            }
        }

        /**
         * Closes the internal thread pool. Should be called when loader is no longer
         * needed
         * or via a shutdown hook if necessary.
         */
        public void shutdown() {
            if (executor != null) {
                executor.shutdownNow();
            }
        }
    }
}
