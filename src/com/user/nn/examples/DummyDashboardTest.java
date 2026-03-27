package com.user.nn.examples;

import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.visualization.TrainingHistory;

import java.util.HashMap;
import java.util.Map;

public class DummyDashboardTest {
    public static void main(String[] args) throws InterruptedException {
        TrainingHistory history = new TrainingHistory();
        System.out.println("Starting Web dashboard...");
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        System.out.println("Dashboard is running at http://localhost:7070");
        System.out.println("Simulating training epochs...");

        dashboard.registerHandler("test_task", (fileName, fileStream, text) -> {
            Map<String, Object> response = new HashMap<>();
            response.put("echo_text", text);
            response.put("file_name_received", fileName);
            response.put("status", "success");
            return response;
        });

        // Simulate 1000 epochs of training so user has time to view realtime updates
        for (int epoch = 1; epoch <= 1000; epoch++) {
            Thread.sleep(1500); // 1.5 seconds per epoch

            // Fake metrics
            float trainLoss = (float) (2.0 * Math.exp(-epoch * 0.1) + Math.random() * 0.1);
            float trainAcc = (float) (1.0 - Math.exp(-epoch * 0.15) - Math.random() * 0.05);

            Map<String, Float> metrics = new HashMap<>();
                metrics.put("train_loss", trainLoss);
            metrics.put("train_acc", trainAcc);

            history.record(epoch, metrics);
            dashboard.broadcastMetrics(epoch, metrics);

            System.out.println("Epoch " + epoch + " - Loss: " + trainLoss + " Acc: " + trainAcc);
        }

        System.out.println("Training done. Keeping server alive for 5 minutes.");
        Thread.sleep(300000); // Keep alive for viewing
        dashboard.stop();
    }
}
