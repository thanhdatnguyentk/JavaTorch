package com.user.nn.utils.visualization.viewers;

import com.user.nn.utils.visualization.Plot;
import com.user.nn.utils.visualization.PlotContext;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * Display plots in a Swing window.
 * Supports real-time updates and interactive viewing.
 * 
 * Example:
 * <pre>
 * LinePlot plot = new LinePlot(x, y, "Data");
 * PlotViewer.show(plot, new PlotContext().title("My Plot"));
 * </pre>
 */
public class PlotViewer {
    
    /**
     * Show plot in a window (blocking until window is closed).
     * 
     * @param plot The plot to display
     * @param context Plot configuration
     */
    public static void show(Plot plot, PlotContext context) {
        JFreeChart chart = plot.toChart(context);
        showChart(chart, context.getTitle(), context.getWidth(), context.getHeight(), true);
    }
    
    /**
     * Show plot with default context.
     */
    public static void show(Plot plot) {
        show(plot, new PlotContext());
    }
    
    /**
     * Show plot in a non-blocking window (continues execution).
     * 
     * @param plot The plot to display
     * @param context Plot configuration
     * @return The JFrame for further manipulation
     */
    public static JFrame showNonBlocking(Plot plot, PlotContext context) {
        JFreeChart chart = plot.toChart(context);
        return showChart(chart, context.getTitle(), context.getWidth(), context.getHeight(), false);
    }
    
    /**
     * Show plot non-blocking with default context.
     */
    public static JFrame showNonBlocking(Plot plot) {
        return showNonBlocking(plot, new PlotContext());
    }
    
    /**
     * Internal method to create and show chart window.
     */
    private static JFrame showChart(JFreeChart chart, String title, int width, int height, boolean blocking) {
        // Create chart panel
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(width, height));
        chartPanel.setMouseWheelEnabled(true); // Enable zoom with mouse wheel
        
        // Create frame
        JFrame frame = new JFrame(title.isEmpty() ? "Plot" : title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        frame.add(chartPanel, BorderLayout.CENTER);
        
        // Add toolbar with save button
        JPanel toolbar = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton saveButton = new JButton("Save as PNG...");
        saveButton.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setDialogTitle("Save as PNG");
            int result = fileChooser.showSaveDialog(frame);
            if (result == JFileChooser.APPROVE_OPTION) {
                try {
                    java.io.File file = fileChooser.getSelectedFile();
                    String path = file.getAbsolutePath();
                    if (!path.toLowerCase().endsWith(".png")) {
                        path += ".png";
                    }
                    org.jfree.chart.ChartUtils.saveChartAsPNG(new java.io.File(path), chart, width, height);
                    JOptionPane.showMessageDialog(frame, "Saved to: " + path, "Success", JOptionPane.INFORMATION_MESSAGE);
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(frame, "Error saving file: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
                }
            }
        });
        toolbar.add(saveButton);
        frame.add(toolbar, BorderLayout.SOUTH);
        
        frame.pack();
        frame.setLocationRelativeTo(null); // Center on screen
        frame.setVisible(true);
        
        if (blocking) {
            // Wait for window to close
            final Object lock = new Object();
            frame.addWindowListener(new WindowAdapter() {
                @Override
                public void windowClosed(WindowEvent e) {
                    synchronized (lock) {
                        lock.notify();
                    }
                }
            });
            
            synchronized (lock) {
                try {
                    lock.wait();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
        
        return frame;
    }
    
    /**
     * Create an updatable plot viewer for real-time plotting.
     * Returns a viewer that can be updated with new data.
     */
    public static UpdatablePlotViewer createUpdatable(Plot initialPlot, PlotContext context) {
        return new UpdatablePlotViewer(initialPlot, context);
    }
    
    /**
     * Updatable plot viewer for real-time updates.
     */
    public static class UpdatablePlotViewer {
        private ChartPanel chartPanel;
        private JFrame frame;
        private PlotContext context;
        
        private UpdatablePlotViewer(Plot initialPlot, PlotContext context) {
            this.context = context;
            JFreeChart chart = initialPlot.toChart(context);
            
            chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(context.getWidth(), context.getHeight()));
            
            frame = new JFrame(context.getTitle().isEmpty() ? "Plot" : context.getTitle());
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.add(chartPanel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        }
        
        /**
         * Update the plot with new data.
         */
        public void update(Plot newPlot) {
            SwingUtilities.invokeLater(() -> {
                JFreeChart newChart = newPlot.toChart(context);
                chartPanel.setChart(newChart);
            });
        }
        
        /**
         * Close the viewer.
         */
        public void close() {
            frame.dispose();
        }
        
        /**
         * Check if the viewer is still open.
         */
        public boolean isOpen() {
            return frame.isVisible();
        }
    }
}
