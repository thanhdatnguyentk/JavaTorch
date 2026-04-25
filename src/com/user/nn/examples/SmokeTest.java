package com.user.nn.examples;

public class SmokeTest {
    public static boolean isEnabled() {
        return System.getProperty("smokeTest") != null;
    }
    
    public static int getEpochs(int defaultEpochs) {
        return isEnabled() ? 1 : defaultEpochs;
    }

    public static int getBatches(int defaultBatches) {
        return isEnabled() ? 1 : defaultBatches;
    }
}
