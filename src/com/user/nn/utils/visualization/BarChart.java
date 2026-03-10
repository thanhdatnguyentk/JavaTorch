package com.user.nn.utils.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import java.awt.Color;

/**
 * Bar chart implementation.
 * Supports horizontal/vertical orientation and grouped bars.
 * 
 * Example:
 * <pre>
 * String[] categories = {"Model A", "Model B", "Model C"};
 * double[] values = {0.85, 0.92, 0.88};
 * BarChart chart = new BarChart(categories, values, "Accuracy");
 * </pre>
 */
public class BarChart implements Plot {
    
    private DefaultCategoryDataset dataset;
    private PlotOrientation orientation = PlotOrientation.VERTICAL;
    
    /**
     * Create a simple bar chart with one series.
     * 
     * @param categories Category names
     * @param values Values for each category
     * @param seriesLabel Label for this series
     */
    public BarChart(String[] categories, double[] values, String seriesLabel) {
        if (categories.length != values.length) {
            throw new IllegalArgumentException("Categories and values must have the same length");
        }
        
        dataset = new DefaultCategoryDataset();
        for (int i = 0; i < categories.length; i++) {
            dataset.addValue(values[i], seriesLabel, categories[i]);
        }
    }
    
    /**
     * Create an empty bar chart.
     */
    public BarChart() {
        dataset = new DefaultCategoryDataset();
    }
    
    /**
     * Add a series to the bar chart (for grouped bars).
     * 
     * @param categories Category names
     * @param values Values for each category
     * @param seriesLabel Label for this series
     */
    public BarChart addSeries(String[] categories, double[] values, String seriesLabel) {
        if (categories.length != values.length) {
            throw new IllegalArgumentException("Categories and values must have the same length");
        }
        
        for (int i = 0; i < categories.length; i++) {
            dataset.addValue(values[i], seriesLabel, categories[i]);
        }
        return this;
    }
    
    /**
     * Set chart orientation.
     * 
     * @param horizontal If true, bars are horizontal; if false, vertical
     */
    public BarChart setHorizontal(boolean horizontal) {
        this.orientation = horizontal ? PlotOrientation.HORIZONTAL : PlotOrientation.VERTICAL;
        return this;
    }
    
    @Override
    public JFreeChart toChart(PlotContext context) {
        // Create chart
        JFreeChart chart = ChartFactory.createBarChart(
            context.getTitle(),
            context.getXlabel(),
            context.getYlabel(),
            dataset,
            orientation,
            context.isShowLegend(),
            true, // tooltips
            false // urls
        );
        
        // Apply context
        applyContext(chart, context);
        
        return chart;
    }
    
    private void applyContext(JFreeChart chart, PlotContext context) {
        chart.setBackgroundPaint(context.getBackgroundColor());
        chart.getTitle().setFont(context.getTitleFont());
        
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(context.getPlotBackgroundColor());
        plot.setDomainGridlinesVisible(context.isShowDomainGrid());
        plot.setRangeGridlinesVisible(context.isShowRangeGrid());
        
        plot.getDomainAxis().setLabelFont(context.getAxisLabelFont());
        plot.getDomainAxis().setTickLabelFont(context.getTickLabelFont());
        plot.getRangeAxis().setLabelFont(context.getAxisLabelFont());
        plot.getRangeAxis().setTickLabelFont(context.getTickLabelFont());
        
        if (context.getYmin() != null && context.getYmax() != null) {
            plot.getRangeAxis().setRange(context.getYmin(), context.getYmax());
        }
        
        chart.setAntiAlias(context.isAntiAlias());
        
        // Set default colors
        Color[] colors = {
            new Color(31, 119, 180),
            new Color(255, 127, 14),
            new Color(44, 160, 44),
            new Color(214, 39, 40),
            new Color(148, 103, 189)
        };
        
        for (int i = 0; i < Math.min(dataset.getRowCount(), colors.length); i++) {
            plot.getRenderer().setSeriesPaint(i, colors[i]);
        }
    }
}
