package com.user.nn.utils;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.ZipInputStream;

/**
 * Utility to download and extract COCO dataset automatically.
 * Downloads validation set by default (smaller than train set).
 */
public class COCODatasetDownloader {

    private static final String COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip";
    private static final String COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip";
    
    /**
     * Downloads COCO validation images and annotations to the specified directory.
     * Uses validation set (5K images, ~1GB) instead of train set (118K images, ~18GB).
     * 
     * @param targetDir Base directory where data/coco will be created
     * @return true if successful, false otherwise
     */
    public static boolean downloadCOCODataset(Path targetDir) {
        try {
            Path cocoDir = targetDir.resolve("data/coco");
            Path val2017Dir = cocoDir.resolve("val2017");
            Path annotationsDir = cocoDir.resolve("annotations");
            
            // Check if already downloaded
            if (Files.exists(val2017Dir) && Files.exists(annotationsDir)) {
                System.out.println("COCO dataset already exists at: " + cocoDir);
                long imageCount = Files.list(val2017Dir).filter(p -> p.toString().endsWith(".jpg")).count();
                if (imageCount > 100) {
                    System.out.println("Found " + imageCount + " validation images. Skipping download.");
                    return true;
                }
            }
            
            Files.createDirectories(cocoDir);
            
            System.out.println("=== COCO Dataset Auto-Download ===");
            System.out.println("Downloading validation set (5K images, ~1GB)");
            System.out.println("This may take 10-30 minutes depending on your connection...");
            System.out.println();
            
            // Download validation images
            Path valZip = cocoDir.resolve("val2017.zip");
            if (!Files.exists(val2017Dir) || Files.list(val2017Dir).count() < 100) {
                System.out.println("Downloading validation images...");
                if (!downloadFile(COCO_VAL_IMAGES_URL, valZip)) {
                    return false;
                }
                System.out.println("Extracting validation images...");
                extractZip(valZip, cocoDir);
                Files.deleteIfExists(valZip);
            } else {
                System.out.println("Validation images already present.");
            }
            
            // Download annotations
            Path annZip = cocoDir.resolve("annotations_trainval2017.zip");
            if (!Files.exists(annotationsDir) || !Files.exists(annotationsDir.resolve("instances_val2017.json"))) {
                System.out.println("Downloading annotations...");
                if (!downloadFile(COCO_ANNOTATIONS_URL, annZip)) {
                    return false;
                }
                System.out.println("Extracting annotations...");
                extractZip(annZip, cocoDir);
                Files.deleteIfExists(annZip);
            } else {
                System.out.println("Annotations already present.");
            }
            
            System.out.println();
            System.out.println("COCO dataset ready at: " + cocoDir);
            System.out.println("Images: " + val2017Dir);
            System.out.println("Annotations: " + annotationsDir.resolve("instances_val2017.json"));
            System.out.println();
            
            return true;
            
        } catch (Exception e) {
            System.err.println("Error downloading COCO dataset: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    private static boolean downloadFile(String urlString, Path destination) {
        try {
            URL url = new URL(urlString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(30000);
            
            int responseCode = conn.getResponseCode();
            if (responseCode != 200) {
                System.err.println("Failed to download: HTTP " + responseCode);
                return false;
            }
            
            long fileSize = conn.getContentLengthLong();
            
            try (InputStream in = new BufferedInputStream(conn.getInputStream());
                 FileOutputStream out = new FileOutputStream(destination.toFile())) {
                
                byte[] buffer = new byte[8192];
                int bytesRead;
                long totalBytesRead = 0;
                long lastPrintTime = System.currentTimeMillis();
                
                while ((bytesRead = in.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                    totalBytesRead += bytesRead;
                    
                    // Print progress every 2 seconds
                    long now = System.currentTimeMillis();
                    if (now - lastPrintTime > 2000) {
                        double percent = fileSize > 0 ? (totalBytesRead * 100.0 / fileSize) : 0;
                        System.out.printf("  Progress: %.1f%% (%.1f MB / %.1f MB)%n",
                                percent,
                                totalBytesRead / (1024.0 * 1024.0),
                                fileSize / (1024.0 * 1024.0));
                        lastPrintTime = now;
                    }
                }
                
                System.out.println("  Download complete: " + destination.getFileName());
                return true;
            }
            
        } catch (Exception e) {
            System.err.println("Download failed: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    private static void extractZip(Path zipFile, Path destDir) {
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile.toFile()))) {
            java.util.zip.ZipEntry entry;
            int count = 0;
            long lastPrintTime = System.currentTimeMillis();
            
            while ((entry = zis.getNextEntry()) != null) {
                Path filePath = destDir.resolve(entry.getName());
                
                if (entry.isDirectory()) {
                    Files.createDirectories(filePath);
                } else {
                    Files.createDirectories(filePath.getParent());
                    try (FileOutputStream fos = new FileOutputStream(filePath.toFile())) {
                        byte[] buffer = new byte[8192];
                        int len;
                        while ((len = zis.read(buffer)) > 0) {
                            fos.write(buffer, 0, len);
                        }
                    }
                }
                
                count++;
                long now = System.currentTimeMillis();
                if (now - lastPrintTime > 2000) {
                    System.out.println("  Extracted " + count + " files...");
                    lastPrintTime = now;
                }
                
                zis.closeEntry();
            }
            
            System.out.println("  Extraction complete: " + count + " files");
            
        } catch (Exception e) {
            System.err.println("Extraction failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Quick test to download a minimal COCO subset for fast testing.
     * Only downloads first 100 images and annotations.
     */
    public static boolean downloadMiniCOCO(Path targetDir, int maxImages) {
        System.out.println("Mini COCO download not implemented yet.");
        System.out.println("Falling back to full validation set download.");
        return downloadCOCODataset(targetDir);
    }
    
    public static void main(String[] args) {
        Path workDir = Paths.get(".").toAbsolutePath().normalize();
        System.out.println("Working directory: " + workDir);
        
        boolean success = downloadCOCODataset(workDir);
        if (success) {
            System.out.println("Dataset ready for training!");
        } else {
            System.err.println("Failed to prepare dataset.");
        }
    }
}
