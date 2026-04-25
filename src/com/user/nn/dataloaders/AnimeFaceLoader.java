package com.user.nn.dataloaders;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
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
 * Output format is NCHW Tensor (Batch, Channels=3, H, W).
 */
public class AnimeFaceLoader {

    private AnimeFaceLoader() {
    }

    /**
     * Load images recursively and normalize to [-1, 1].
     */
    public static Tensor loadImagesToTensor(File root, int imageSize, int maxImages) throws IOException {
        return loadImagesToTensor(root, imageSize, maxImages, true);
    }

    /**
     * Load images recursively and return NCHW Tensor.
     *
     * If normalizeToMinusOneOne=true, pixel range is [-1, 1].
     * Otherwise, pixel range is [0, 1].
     */
    public static Tensor loadImagesToTensor(File root, int imageSize, int maxImages, boolean normalizeToMinusOneOne)
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

        int count = imageFiles.size();
        int imageDim = imageSize * imageSize * 3;
        Tensor batchTensor = new Tensor(count, 3, imageSize, imageSize);
        
        for (int i = 0; i < count; i++) {
            loadImageToNCHWTensor(imageFiles.get(i), batchTensor, i, imageSize, normalizeToMinusOneOne);
            if ((i + 1) % 500 == 0 || (i + 1) == count) {
                System.out.printf(Locale.US, "Loaded %d/%d images\r", i + 1, count);
            }
        }
        if (count > 0) {
            System.out.println();
        }
        return batchTensor;
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

    private static void loadImageToNCHWTensor(File imageFile, Tensor batchTensor, int bIndex, int imageSize,
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

        int hw = imageSize * imageSize;
        int sampleOffset = bIndex * 3 * hw;

        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                int rgb = resized.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gch = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int idx = y * imageSize + x;

                if (normalizeToMinusOneOne) {
                    batchTensor.data[sampleOffset + idx] = (r / 127.5f) - 1.0f;
                    batchTensor.data[sampleOffset + hw + idx] = (gch / 127.5f) - 1.0f;
                    batchTensor.data[sampleOffset + 2 * hw + idx] = (b / 127.5f) - 1.0f;
                } else {
                    batchTensor.data[sampleOffset + idx] = r / 255.0f;
                    batchTensor.data[sampleOffset + hw + idx] = gch / 255.0f;
                    batchTensor.data[sampleOffset + 2 * hw + idx] = b / 255.0f;
                }
            }
        }
    }
}
