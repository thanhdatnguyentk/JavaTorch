package com.user.nn.utils.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.SymbolAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.data.xy.DefaultXYZDataset;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.LookupPaintScale;

import java.awt.Color;

/**
 * Heatmap plot implementation.
 * Displays a 2D matrix with color-coded values.
 * 
 * Example:
 * <pre>
 * double[][] matrix = {
 *     {1.0, 2.0, 3.0},
 *     {4.0, 5.0, 6.0},
 *     {7.0, 8.0, 9.0}
 * };
 * HeatmapPlot heatmap = new HeatmapPlot(matrix);
 * heatmap.setTickLabels(new String[]{"A", "B", "C"}, new String[]{"X", "Y", "Z"});
 * </pre>
 */
public class HeatmapPlot implements Plot {
    
    private double[][] matrix;
    private String[] xLabels = null;
    private String[] yLabels = null;
    private String colormap = "viridis";
    private boolean annotations = false;
    
    /**
     * Create a heatmap from a 2D matrix.
     * 
     * @param matrix 2D array of values (rows x cols)
     */
    public HeatmapPlot(double[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            throw new IllegalArgumentException("Matrix cannot be null or empty");
        }
        this.matrix = matrix;
    }
    
    /**
     * Set tick labels for axes.
     * 
     * @param xLabels Labels for x-axis (columns)
     * @param yLabels Labels for y-axis (rows)
     */
    public HeatmapPlot setTickLabels(String[] xLabels, String[] yLabels) {
        this.xLabels = xLabels;
        this.yLabels = yLabels;
        return this;
    }
    
    /**
     * Set colormap.
     * Supported: "viridis", "plasma", "coolwarm", "RdBu"
     */
    public HeatmapPlot setColormap(String colormap) {
        this.colormap = colormap;
        return this;
    }
    
    /**
     * Enable/disable value annotations in cells.
     */
    public HeatmapPlot setAnnotations(boolean annotations) {
        this.annotations = annotations;
        return this;
    }
    
    @Override
    public JFreeChart toChart(PlotContext context) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        
        // Find min/max for color scale
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double[] row : matrix) {
            for (double val : row) {
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }
        
        // Create dataset
        DefaultXYZDataset dataset = new DefaultXYZDataset();
        double[][] data = new double[3][rows * cols];
        
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[0][idx] = j;      // X
                data[1][idx] = rows - 1 - i; // Y (flip for matrix convention)
                data[2][idx] = matrix[i][j];  // Z (value)
                idx++;
            }
        }
        dataset.addSeries("data", data);
        
        // Create chart
        NumberAxis xAxis = new NumberAxis(context.getXlabel());
        NumberAxis yAxis = new NumberAxis(context.getYlabel());
        
        // If labels provided, use SymbolAxis
        if (xLabels != null && xLabels.length == cols) {
            double[] xTicks = new double[cols];
            for (int i = 0; i < cols; i++) xTicks[i] = i;
            xAxis = new SymbolAxis(context.getXlabel(), xLabels);
        }
        
        if (yLabels != null && yLabels.length == rows) {
            String[] flippedYLabels = new String[rows];
            for (int i = 0; i < rows; i++) {
                flippedYLabels[i] = yLabels[rows - 1 - i];
            }
            yAxis = new SymbolAxis(context.getYlabel(), flippedYLabels);
        }
        
        XYBlockRenderer renderer = new XYBlockRenderer();
        renderer.setBlockWidth(1.0);
        renderer.setBlockHeight(1.0);
        
        // Create color scale
        PaintScale paintScale = createPaintScale(min, max);
        renderer.setPaintScale(paintScale);
        
        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
        plot.setOrientation(PlotOrientation.VERTICAL);
        
        JFreeChart chart = new JFreeChart(context.getTitle(), plot);
        
        // Apply context
        applyContext(chart, context);
        
        return chart;
    }
    
    /**
     * Create a paint scale for the colormap.
     */
    private PaintScale createPaintScale(double min, double max) {
        LookupPaintScale scale = new LookupPaintScale(min, max, Color.WHITE);
        
        int steps = 256;
        double range = max - min;
        
        for (int i = 0; i < steps; i++) {
            double value = min + (range * i / steps);
            Color color = getColormapColor(i / (double) steps);
            scale.add(value, color);
        }
        
        return scale;
    }
    
    /**
     * Get color from colormap at position t (0-1).
     */
    private Color getColormapColor(double t) {
        switch (colormap.toLowerCase()) {
            case "plasma":
                return plasmaColormap(t);
            case "coolwarm":
                return coolwarmColormap(t);
            case "rdbu":
                return rdbuColormap(t);
            default: // viridis
                return viridisColormap(t);
        }
    }
    
    /**
     * Viridis colormap (perceptually uniform).
     */
    private Color viridisColormap(double t) {
        // Simplified viridis approximation
        double r = Math.max(0, Math.min(1, 0.267 + 0.005 * t + 0.322 * Math.pow(t, 2)));
        double g = Math.max(0, Math.min(1, 0.005 + 0.570 * t + 0.115 * Math.pow(t, 2)));
        double b = Math.max(0, Math.min(1, 0.329 + 0.880 * t - 0.560 * Math.pow(t, 2)));
        return new Color((float) r, (float) g, (float) b);
    }
    
    /**
     * Plasma colormap.
     */
    private Color plasmaColormap(double t) {
        double r = Math.max(0, Math.min(1, 0.050 + 1.080 * Math.pow(t, 1.5)));
        double g = Math.max(0, Math.min(1, 0.030 + 0.720 * Math.pow(t, 2) - 0.300 * Math.pow(t, 3)));
        double b = Math.max(0, Math.min(1, 0.527 + 0.560 * t - 1.100 * Math.pow(t, 2)));
        return new Color((float) r, (float) g, (float) b);
    }
    
    /**
     * Coolwarm diverging colormap.
     */
    private Color coolwarmColormap(double t) {
        if (t < 0.5) {
            // Cool (blue)
            double s = t * 2;
            return new Color((float) (0.230 + 0.300 * s), (float) (0.299 + 0.400 * s), (float) (0.754 - 0.200 * s));
        } else {
            // Warm (red)
            double s = (t - 0.5) * 2;
            return new Color((float) (0.530 + 0.400 * s), (float) (0.699 - 0.400 * s), (float) (0.554 - 0.400 * s));
        }
    }
    
    /**
     * RdBu diverging colormap (red-blue).
     */
    private Color rdbuColormap(double t) {
        if (t < 0.5) {
            // Blue side
            double s = t * 2;
            return new Color((float) (0.020 + 0.200 * s), (float) (0.188 + 0.400 * s), (float) (0.380 + 0.520 * s));
        } else {
            // Red side
            double s = (t - 0.5) * 2;
            return new Color((float) (0.220 + 0.650 * s), (float) (0.588 - 0.480 * s), (float) (0.900 - 0.770 * s));
        }
    }
    
    private void applyContext(JFreeChart chart, PlotContext context) {
        chart.setBackgroundPaint(context.getBackgroundColor());
        chart.getTitle().setFont(context.getTitleFont());
        
        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(context.getPlotBackgroundColor());
        
        plot.getDomainAxis().setLabelFont(context.getAxisLabelFont());
        plot.getDomainAxis().setTickLabelFont(context.getTickLabelFont());
        plot.getRangeAxis().setLabelFont(context.getAxisLabelFont());
        plot.getRangeAxis().setTickLabelFont(context.getTickLabelFont());
        
        chart.setAntiAlias(context.isAntiAlias());
    }
}
