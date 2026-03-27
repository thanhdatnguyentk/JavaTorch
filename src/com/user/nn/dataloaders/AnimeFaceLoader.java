package com.user.nn.dataloaders;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import javax.imageio.ImageIO;

/**
 * Loader for Anime Face datasets stored as image folders.
 *
 * Output format is CHW flattened vector (C=3, H=imageSize, W=imageSize):
 * [R_plane | G_plane | B_plane].
 */
public class AnimeFaceLoader {

    private AnimeFaceLoader() {
    }

    /**
     * Load images recursively and normalize to [-1, 1].
     */
    public static float[][] loadImages(File root, int imageSize, int maxImages) throws IOException {
        return loadImages(root, imageSize, maxImages, true);
    }

    /**
     * Load images recursively and return CHW vectors.
     *
     * If normalizeToMinusOneOne=true, pixel range is [-1, 1].
     * Otherwise, pixel range is [0, 1].
     */
    public static float[][] loadImages(File root, int imageSize, int maxImages, boolean normalizeToMinusOneOne)
            throws IOException {
        if (root == null || !root.exists() || !root.isDirectory()) {
            throw new IllegalArgumentException("Dataset folder not found: " + (root == null ? "null" : root.getAbsolutePath()));
        }
        if (imageSize <= 0) {
            throw new IllegalArgumentException("imageSize must be > 0");
        }

        List<File> imageFiles = new ArrayList<>();
        collectImageFiles(root, imageFiles);
        if (maxImages > 0 && imageFiles.size() > maxImages) {
            imageFiles = imageFiles.subList(0, maxImages);
        }

        int imageDim = imageSize * imageSize * 3;
        float[][] data = new float[imageFiles.size()][];
        for (int i = 0; i < imageFiles.size(); i++) {
            data[i] = loadImageAsChwVector(imageFiles.get(i), imageSize, imageDim, normalizeToMinusOneOne);
            if ((i + 1) % 500 == 0 || (i + 1) == imageFiles.size()) {
                System.out.printf(Locale.US, "Loaded %d/%d images\r", i + 1, imageFiles.size());
            }
        }
        if (!imageFiles.isEmpty()) {
            System.out.println();
        }
        return data;
    }

    public static List<File> listImageFiles(File root) {
        List<File> out = new ArrayList<>();
        collectImageFiles(root, out);
        return out;
    }

    private static void collectImageFiles(File dir, List<File> out) {
        File[] children = dir.listFiles();
        if (children == null) {
            return;
        }
        for (File f : children) {
            if (f.isDirectory()) {
                collectImageFiles(f, out);
                continue;
            }
            String n = f.getName().toLowerCase(Locale.ROOT);
            if (n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png") || n.endsWith(".webp")) {
                out.add(f);
            }
        }
    }

    private static float[] loadImageAsChwVector(File imageFile, int imageSize, int imageDim,
            boolean normalizeToMinusOneOne) throws IOException {
        BufferedImage src = ImageIO.read(imageFile);
        if (src == null) {
            throw new IOException("Unsupported image format: " + imageFile.getAbsolutePath());
        }

        BufferedImage resized = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, imageSize, imageSize, null);
        g.dispose();

        float[] out = new float[imageDim];
        int hw = imageSize * imageSize;

        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                int rgb = resized.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gch = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int idx = y * imageSize + x;

                if (normalizeToMinusOneOne) {
                    out[idx] = (r / 127.5f) - 1.0f;
                    out[hw + idx] = (gch / 127.5f) - 1.0f;
                    out[2 * hw + idx] = (b / 127.5f) - 1.0f;
                } else {
                    out[idx] = r / 255.0f;
                    out[hw + idx] = gch / 255.0f;
                    out[2 * hw + idx] = b / 255.0f;
                }
            }
        }

        return out;
    }
}
