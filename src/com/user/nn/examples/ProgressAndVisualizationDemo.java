package com.user.nn.examples;

import com.user.nn.utils.progress.ProgressBar;
import com.user.nn.utils.visualization.*;
import com.user.nn.utils.visualization.exporters.FileExporter;
import com.user.nn.utils.visualization.exporters.HTMLExporter;
import com.user.nn.utils.visualization.viewers.PlotViewer;

import java.util.HashMap;
import java.util.Map;

/**
 * Comprehensive demo of progress bar and visualization features.
 * 
 * This example demonstrates:
 * - Progress bars with metrics
 * - Line plots, scatter plots, histograms, heatmaps, bar charts
 * - Training history tracking
 * - File export (PNG, SVG, HTML)
 * - Interactive viewing
 * - ML-specific visualizations
 */
public class ProgressAndVisualizationDemo {
    
    public static void main(String[] args) {
        System.out.println("=== ML Framework Progress & Visualization Demo ===\n");
        
        // Demo 1: Progress Bar
        demoProgressBar();
        
        // Demo 2: Basic Plots
        demoBasicPlots();
        
        // Demo 3: Training History
        demoTrainingHistory();
        
        // Demo 4: ML-Specific Visualizations
        demoMLVisualizations();
        
        System.out.println("\n=== All demos completed! ===");
        System.out.println("Check the 'demo_outputs' directory for saved plots.");
    }
    
    /**
     * Demo 1: Progress bar with metrics
     */
    private static void demoProgressBar() {
        System.out.println("Demo 1: Progress Bar");
        
        int total = 100;
        ProgressBar bar = new ProgressBar(total, "Processing");
        
        for (int i = 0; i < total; i++) {
            // Simulate work
            try {
                Thread.sleep(30);
            } catch (InterruptedException e) {
                break;
            }
            
            // Update metrics
            float loss = (float) (1.0 - i / (double) total + 0.1 * Math.random());
            float acc = (float) (i / (double) total + 0.05 * Math.random());
            
            bar.setPostfix("loss", loss);
            bar.setPostfix("acc", acc);
            bar.update(1);
        }
        
        bar.close();
        System.out.println("");
    }
    
    /**
     * Demo 2: Basic plot types
     */
    private static void demoBasicPlots() {
        System.out.println("\nDemo 2: Basic Plots");
        
        try {
            // 1. Line Plot
            double[] x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            double[] y1 = {2, 4, 3, 5, 4, 6, 5, 7, 6, 8};
            double[] y2 = {1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
            
            LinePlot linePlot = new LinePlot(x, y1, "Series 1");
            linePlot.addSeries(x, y2, "Series 2");
            linePlot.setMarker(0, true, "circle");
            
            PlotContext lineCtx = new PlotContext()
                .title("Line Plot Example")
                .xlabel("X Axis")
                .ylabel("Y Axis")
                .grid(true);
            
            FileExporter.savePNG(linePlot, lineCtx, "demo_outputs/line_plot.png");
            System.out.println("✓ Saved line_plot.png");
            
            // 2. Scatter Plot
            double[] scatterX = {1, 2, 3, 4, 5, 6, 7, 8};
            double[] scatterY = {2, 4, 1, 5, 3, 6, 2, 7};
            
            ScatterPlot scatterPlot = new ScatterPlot(scatterX, scatterY, "Data Points");
            scatterPlot.setAlpha(0.7);
            scatterPlot.setBaseSize(10.0);
            
            PlotContext scatterCtx = new PlotContext()
                .title("Scatter Plot Example")
                .xlabel("Feature 1")
                .ylabel("Feature 2");
            
            FileExporter.savePNG(scatterPlot, scatterCtx, "demo_outputs/scatter_plot.png");
            System.out.println("✓ Saved scatter_plot.png");
            
            // 3. Histogram
            double[] data = new double[1000];
            for (int i = 0; i < data.length; i++) {
                data[i] = Math.random() * 10 - 5 + (Math.random() - 0.5) * 2;
            }
            
            Histogram histogram = new Histogram(data, "Normal Distribution");
            histogram.setBins(30);
            
            PlotContext histCtx = new PlotContext()
                .title("Histogram Example")
                .xlabel("Value")
                .ylabel("Frequency");
            
            FileExporter.savePNG(histogram, histCtx, "demo_outputs/histogram.png");
            System.out.println("✓ Saved histogram.png");
            
            // 4. Heatmap
            double[][] matrix = {
                {1.0, 0.8, 0.3, 0.1},
                {0.8, 1.0, 0.5, 0.2},
                {0.3, 0.5, 1.0, 0.7},
                {0.1, 0.2, 0.7, 1.0}
            };
            String[] labels = {"A", "B", "C", "D"};
            
            HeatmapPlot heatmap = new HeatmapPlot(matrix);
            heatmap.setTickLabels(labels, labels);
            heatmap.setColormap("viridis");
            
            PlotContext heatCtx = new PlotContext()
                .title("Heatmap Example")
                .xlabel("Feature")
                .ylabel("Feature");
            
            FileExporter.savePNG(heatmap, heatCtx, "demo_outputs/heatmap.png");
            System.out.println("✓ Saved heatmap.png");
            
            // 5. Bar Chart
            String[] categories = {"Model A", "Model B", "Model C", "Model D"};
            double[] values = {0.85, 0.92, 0.88, 0.91};
            
            BarChart barChart = new BarChart(categories, values, "Accuracy");
            
            PlotContext barCtx = new PlotContext()
                .title("Model Comparison")
                .xlabel("Model")
                .ylabel("Accuracy")
                .ylim(0.8, 1.0);
            
            FileExporter.savePNG(barChart, barCtx, "demo_outputs/bar_chart.png");
            System.out.println("✓ Saved bar_chart.png");
            
            // Save one as HTML
            HTMLExporter.saveHTML(linePlot, lineCtx, "demo_outputs/interactive_plot.html");
            System.out.println("✓ Saved interactive_plot.html");
            
        } catch (Exception e) {
            System.err.println("Error in basic plots demo: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Demo 3: Training history tracking
     */
    private static void demoTrainingHistory() {
        System.out.println("\nDemo 3: Training History");
        
        try {
            TrainingHistory history = new TrainingHistory();
        // Simulate training
            System.out.print("Simulating training");
            for (int epoch = 0; epoch < 50; epoch++) {
                if (epoch % 10 == 0) System.out.print(".");
                
                float trainLoss = (float) (1.0 / (epoch + 1) + 0.1 * Math.random());
                float valLoss = (float) (1.2 / (epoch + 1) + 0.1 * Math.random());
                float trainAcc = (float) (0.5 + 0.4 * epoch / 50.0 + 0.05 * Math.random());
                float valAcc = (float) (0.5 + 0.38 * epoch / 50.0 + 0.05 * Math.random());
                
                Map<String, Float> metrics = new HashMap<>();
                metrics.put("train_loss", trainLoss);
                metrics.put("val_loss", valLoss);
                metrics.put("train_acc", trainAcc);
                metrics.put("val_acc", valAcc);
                
                history.record(epoch, metrics);
            }
            System.out.println(" done!");
            
            // Plot training curves
            Plot trainingPlot = history.plot();
            PlotContext trainCtx = new PlotContext()
                .title("Training History")
                .xlabel("Epoch")
                .ylabel("Value")
                .grid(true);
            
            FileExporter.savePNG(trainingPlot, trainCtx, "demo_outputs/training_curves.png");
            System.out.println("✓ Saved training_curves.png");
            
            // Save CSV
            history.saveCSV("demo_outputs/training_history.csv");
            System.out.println("✓ Saved training_history.csv");
            
            // Print best results
            System.out.println("  Best val_acc: " + history.getMax("val_acc") + 
                             " at epoch " + history.getMaxEpoch("val_acc"));
            
        } catch (Exception e) {
            System.err.println("Error in training history demo: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Demo 4: ML-specific visualizations
     */
    private static void demoMLVisualizations() {
        System.out.println("\nDemo 4: ML-Specific Visualizations");
        
        try {
            // 1. Confusion Matrix
            int[][] confusionMatrix = {
                {50, 5, 2},
                {3, 48, 4},
                {1, 2, 52}
            };
            String[] classLabels = {"Cat", "Dog", "Bird"};
            
            HeatmapPlot cmPlot = MLViz.plotConfusionMatrix(confusionMatrix, classLabels);
            PlotContext cmCtx = new PlotContext()
                .title("Confusion Matrix")
                .xlabel("Predicted")
                .ylabel("Actual");
            
            FileExporter.savePNG(cmPlot, cmCtx, "demo_outputs/confusion_matrix.png");
            System.out.println("✓ Saved confusion_matrix.png");
            
            // 2. Learning Rate Schedule (simulated)
            double[] epochs = new double[100];
            double[] lrs = new double[100];
            for (int i = 0; i < 100; i++) {
                epochs[i] = i;
                // Cosine annealing
                lrs[i] = 0.001 + 0.009 * (1 + Math.cos(Math.PI * i / 100)) / 2;
            }
            
            LinePlot lrPlot = MLViz.plotLearningRateSchedule(epochs, lrs);
            PlotContext lrCtx = new PlotContext()
                .title("Learning Rate Schedule")
                .xlabel("Epoch")
                .ylabel("Learning Rate")
                .grid(true);
            
            FileExporter.savePNG(lrPlot, lrCtx, "demo_outputs/lr_schedule.png");
            System.out.println("✓ Saved lr_schedule.png");
            
        } catch (Exception e) {
            System.err.println("Error in ML visualizations demo: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
