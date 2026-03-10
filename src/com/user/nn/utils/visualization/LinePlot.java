package com.user.nn.utils.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Line plot implementation.
 * Supports multiple series with different styles and colors.
 * 
 * Example:
 * <pre>
 * double[] x = {1, 2, 3, 4, 5};
 * double[] y1 = {1, 4, 2, 5, 3};
 * double[] y2 = {2, 3, 4, 3, 5};
 * 
 * LinePlot plot = new LinePlot(x, y1, "Series 1");
 * plot.addSeries(x, y2, "Series 2");
 * </pre>
 */
public class LinePlot implements Plot {
    
    private final List<SeriesData> seriesList;
    
    /**
     * Create an empty line plot.
     */
    public LinePlot() {
        this.seriesList = new ArrayList<>();
    }
    
    /**
     * Create a line plot with one series.
     * 
     * @param x X-axis values
     * @param y Y-axis values
     * @param label Series label
     */
    public LinePlot(double[] x, double[] y, String label) {
        this();
        addSeries(x, y, label);
    }
    
    /**
     * Create a line plot with auto-generated x values (0, 1, 2, ...).
     * 
     * @param y Y-axis values
     * @param label Series label
     */
    public LinePlot(double[] y, String label) {
        this();
        double[] x = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            x[i] = i;
        }
        addSeries(x, y, label);
    }
    
    /**
     * Add a series to the plot.
     * 
     * @param x X-axis values
     * @param y Y-axis values
     * @param label Series label
     */
    public LinePlot addSeries(double[] x, double[] y, String label) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must have the same length");
        }
        seriesList.add(new SeriesData(x, y, label));
        return this;
    }
    
    /**
     * Add a series with auto-generated x values.
     * 
     * @param y Y-axis values
     * @param label Series label
     */
    public LinePlot addSeries(double[] y, String label) {
        double[] x = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            x[i] = i;
        }
        return addSeries(x, y, label);
    }
    
    /**
     * Set line style for a specific series.
     * 
     * @param seriesIndex Series index (0-based)
     * @param style Line style: "solid", "dashed", "dotted"
     */
    public LinePlot setLineStyle(int seriesIndex, String style) {
        if (seriesIndex >= 0 && seriesIndex < seriesList.size()) {
            seriesList.get(seriesIndex).lineStyle = style;
        }
        return this;
    }
    
    /**
     * Set marker style for a specific series.
     * 
     * @param seriesIndex Series index (0-based)
     * @param showMarker Whether to show markers
     * @param markerStyle Marker style: "circle", "square", "triangle", "diamond"
     */
    public LinePlot setMarker(int seriesIndex, boolean showMarker, String markerStyle) {
        if (seriesIndex >= 0 && seriesIndex < seriesList.size()) {
            SeriesData series = seriesList.get(seriesIndex);
            series.showMarker = showMarker;
            series.markerStyle = markerStyle;
        }
        return this;
    }
    
    /**
     * Set color for a specific series.
     * 
     * @param seriesIndex Series index (0-based)
     * @param color Color
     */
    public LinePlot setColor(int seriesIndex, Color color) {
        if (seriesIndex >= 0 && seriesIndex < seriesList.size()) {
            seriesList.get(seriesIndex).color = color;
        }
        return this;
    }
    
    @Override
    public JFreeChart toChart(PlotContext context) {
        // Create dataset
        XYSeriesCollection dataset = new XYSeriesCollection();
        
        for (SeriesData series : seriesList) {
            XYSeries xySeries = new XYSeries(series.label);
            for (int i = 0; i < series.x.length; i++) {
                xySeries.add(series.x[i], series.y[i]);
            }
            dataset.addSeries(xySeries);
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
            context.getTitle(),
            context.getXlabel(),
            context.getYlabel(),
            dataset,
            PlotOrientation.VERTICAL,
            context.isShowLegend(),
            true, // tooltips
            false // urls
        );
        
        // Apply context settings
        applyContext(chart, context);
        
        return chart;
    }
    
    /**
     * Apply context settings to the chart.
     */
    private void applyContext(JFreeChart chart, PlotContext context) {
        // Background
        chart.setBackgroundPaint(context.getBackgroundColor());
        
        // Title font
        chart.getTitle().setFont(context.getTitleFont());
        
        // Get plot
        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(context.getPlotBackgroundColor());
        
        // Grid
        plot.setDomainGridlinesVisible(context.isShowDomainGrid());
        plot.setRangeGridlinesVisible(context.isShowRangeGrid());
        
        // Axis fonts
        plot.getDomainAxis().setLabelFont(context.getAxisLabelFont());
        plot.getDomainAxis().setTickLabelFont(context.getTickLabelFont());
        plot.getRangeAxis().setLabelFont(context.getAxisLabelFont());
        plot.getRangeAxis().setTickLabelFont(context.getTickLabelFont());
        
        // Axis limits
        if (context.getXmin() != null && context.getXmax() != null) {
            plot.getDomainAxis().setRange(context.getXmin(), context.getXmax());
        }
        if (context.getYmin() != null && context.getYmax() != null) {
            plot.getRangeAxis().setRange(context.getYmin(), context.getYmax());
        }
        
        // Anti-aliasing
        chart.setAntiAlias(context.isAntiAlias());
        
        // Renderer
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        
        // Default colors if not specified
        Color[] defaultColors = {
            new Color(31, 119, 180), // blue
            new Color(255, 127, 14), // orange
            new Color(44, 160, 44),  // green
            new Color(214, 39, 40),  // red
            new Color(148, 103, 189), // purple
            new Color(140, 86, 75),  // brown
            new Color(227, 119, 194), // pink
            new Color(127, 127, 127), // gray
            new Color(188, 189, 34),  // olive
            new Color(23, 190, 207)   // cyan
        };
        
        for (int i = 0; i < seriesList.size(); i++) {
            SeriesData series = seriesList.get(i);
            
            // Color
            Color color = series.color != null ? series.color : defaultColors[i % defaultColors.length];
            renderer.setSeriesPaint(i, color);
            
            // Line style
            Stroke stroke = getStroke(series.lineStyle);
            renderer.setSeriesStroke(i, stroke);
            
            // Markers
            renderer.setSeriesShapesVisible(i, series.showMarker);
            if (series.showMarker) {
                Shape shape = getMarkerShape(series.markerStyle);
                renderer.setSeriesShape(i, shape);
            }
            
            // Always show lines
            renderer.setSeriesLinesVisible(i, true);
        }
        
        plot.setRenderer(renderer);
    }
    
    /**
     * Get stroke for line style.
     */
    private Stroke getStroke(String style) {
        switch (style.toLowerCase()) {
            case "dashed":
                return new BasicStroke(2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                                      1.0f, new float[]{10.0f, 6.0f}, 0.0f);
            case "dotted":
                return new BasicStroke(2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                                      1.0f, new float[]{2.0f, 6.0f}, 0.0f);
            default: // solid
                return new BasicStroke(2.0f);
        }
    }
    
    /**
     * Get shape for marker style.
     */
    private Shape getMarkerShape(String style) {
        int size = 6;
        switch (style.toLowerCase()) {
            case "square":
                return new Rectangle(-size/2, -size/2, size, size);
            case "triangle":
                int[] xPoints = {0, size/2, -size/2};
                int[] yPoints = {-size/2, size/2, size/2};
                return new Polygon(xPoints, yPoints, 3);
            case "diamond":
                int[] xDiamond = {0, size/2, 0, -size/2};
                int[] yDiamond = {-size/2, 0, size/2, 0};
                return new Polygon(xDiamond, yDiamond, 4);
            default: // circle
                return new java.awt.geom.Ellipse2D.Double(-size/2.0, -size/2.0, size, size);
        }
    }
    
    /**
     * Internal class to hold series data.
     */
    private static class SeriesData {
        double[] x;
        double[] y;
        String label;
        String lineStyle = "solid";
        boolean showMarker = false;
        String markerStyle = "circle";
        Color color = null;
        
        SeriesData(double[] x, double[] y, String label) {
            this.x = x;
            this.y = y;
            this.label = label;
        }
    }
}
