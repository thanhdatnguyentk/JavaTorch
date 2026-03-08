package com.user.nn.dataloaders;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.zip.GZIPInputStream;

/**
 * Loader for CIFAR-10 binary format.
 * Downloads the .tar.gz dataset, unpacks binary batches, and loads images +
 * labels.
 */
public class Cifar10Loader {

    public static final String DATA_DIR = "data/cifar-10/";
    public static final String TAR_FILE = DATA_DIR + "cifar-10-binary.tar.gz";

    public static void prepareData() throws Exception {
        File dir = new File(DATA_DIR);
        if (!dir.exists()) dir.mkdirs();

        File tarFile = new File(TAR_FILE);
        if (!tarFile.exists() || tarFile.length() == 0) {
            String url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
            System.out.println("Downloading " + url + " ...");
            try (InputStream in = new URL(url).openStream();
                 FileOutputStream out = new FileOutputStream(tarFile)) {
                byte[] buf = new byte[8192];
                int n;
                while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            }
            System.out.println("Downloaded to " + tarFile.getPath());
        }

        File train1 = new File(DATA_DIR + "cifar-10-batches-bin/data_batch_1.bin");
        if (!train1.exists()) {
            System.out.println("Extracting via tar -xzf...");
            ProcessBuilder pb = new ProcessBuilder("tar", "-xzf", "cifar-10-binary.tar.gz");
            pb.directory(new File(DATA_DIR));
            if (pb.start().waitFor() != 0) {
                System.out.println("Warning: tar extraction failed or not available.");
            }
        }
    }

    // Each bin file contains 10,000 images, each image is 3073 bytes (1 byte label + 3072 bytes pixel data)
    public static Object[] loadBatch(String filename) throws IOException {
        int numImages = 10000;
        int imageSize = 3072;
        float[][] images = new float[numImages][imageSize];
        int[] labels = new int[numImages];

        File file = new File(DATA_DIR + "cifar-10-batches-bin/" + filename);
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file))) {
            for (int i = 0; i < numImages; i++) {
                int labelByte = bis.read();
                if (labelByte == -1) throw new EOFException("Unexpected EOF at image " + i);
                labels[i] = labelByte;
                for (int j = 0; j < imageSize; j++) {
                    int pixelByte = bis.read();
                    if (pixelByte == -1) throw new EOFException("Truncated image data at image " + i + ", pixel " + j);
                    images[i][j] = pixelByte / 255.0f;
                }
            }
        }
        return new Object[]{images, labels};
    }
}
