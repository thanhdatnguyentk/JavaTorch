package com.user.nn.dataloaders;

import java.io.*;
import java.net.URL;
import java.util.zip.GZIPInputStream;

/**
 * Loader for MNIST / Fashion-MNIST IDX binary format.
 * Downloads .gz files, decompresses, and parses images + labels.
 */
public class MnistLoader {

    /**
     * Load images from IDX3 file (gzipped). Returns float[numImages][rows*cols],
     * normalized to [0,1].
     */
    public static float[][] loadImages(File gzFile) throws IOException {
        try (DataInputStream dis = new DataInputStream(new GZIPInputStream(new FileInputStream(gzFile)))) {
            int magic = dis.readInt();
            if (magic != 2051)
                throw new IOException("Invalid image file magic: " + magic);
            int numImages = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();
            int size = rows * cols;
            float[][] images = new float[numImages][size];
            for (int i = 0; i < numImages; i++) {
                for (int j = 0; j < size; j++) {
                    images[i][j] = (dis.readUnsignedByte() & 0xFF) / 255.0f;
                }
            }
            return images;
        }
    }

    /** Load labels from IDX1 file (gzipped). Returns int[numLabels]. */
    public static int[] loadLabels(File gzFile) throws IOException {
        try (DataInputStream dis = new DataInputStream(new GZIPInputStream(new FileInputStream(gzFile)))) {
            int magic = dis.readInt();
            if (magic != 2049)
                throw new IOException("Invalid label file magic: " + magic);
            int numLabels = dis.readInt();
            int[] labels = new int[numLabels];
            for (int i = 0; i < numLabels; i++) {
                labels[i] = dis.readUnsignedByte() & 0xFF;
            }
            return labels;
        }
    }

    /** Download file from URL if not already present. */
    public static void downloadIfMissing(String urlStr, File dest) throws IOException {
        if (dest.exists() && dest.length() > 0)
            return;
        dest.getParentFile().mkdirs();
        System.out.println("Downloading " + urlStr + " ...");
        try (InputStream in = new URL(urlStr).openStream();
                FileOutputStream out = new FileOutputStream(dest)) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0)
                out.write(buf, 0, n);
        }
        System.out.println("Saved to " + dest.getPath());
    }
}
