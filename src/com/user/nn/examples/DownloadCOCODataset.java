package com.user.nn.examples;

import com.user.nn.utils.COCODatasetDownloader;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Quick test to verify COCO dataset download functionality.
 * Run this first to download the dataset, then run TrainYOLOCoco.
 */
public class DownloadCOCODataset {
    public static void main(String[] args) {
        System.out.println("=== COCO Dataset Downloader ===");
        System.out.println("This will download COCO validation set (~1GB, 5000 images)");
        System.out.println("Download may take 10-30 minutes depending on your internet speed.");
        System.out.println();
        
        Path workDir = Paths.get(".").toAbsolutePath().normalize();
        System.out.println("Working directory: " + workDir);
        System.out.println();
        
        System.out.println("Starting download...");
        boolean success = COCODatasetDownloader.downloadCOCODataset(workDir);
        
        System.out.println();
        if (success) {
            System.out.println("✓ Dataset download complete!");
            System.out.println("You can now run: ./gradlew :examples:run");
            System.out.println("Or: java com.user.nn.examples.TrainYOLOCoco");
        } else {
            System.err.println("✗ Dataset download failed.");
            System.err.println("Please check your internet connection and try again.");
            System.err.println("Or download manually from: http://cocodataset.org/");
        }
    }
}
