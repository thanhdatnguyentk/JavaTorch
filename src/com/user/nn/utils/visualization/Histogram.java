package com.user.nn.utils.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.statistics.HistogramDataset;

/**
 * Histogram plot implementation.
 * Supports automatic binning algorithms and overlaid histograms.
 * 
 * Example:
 * <pre>
 * double[] data = {1.1, 2.3, 1.5, 2.1, 3.4, 2.8, 1.9}; 
 * Histogram hist = new Histogram(data, 10, "Data Distribution");
 * </pre>
 */
public class Histogram implements Plot {
    
    private double[][] dataSeries;
    private String[] labels;
    private int bins;
    private Double rangeMin = null;
    private Double rangeMax = null;
    private boolean normalize = false;
    
    /**
     * Create a histogram with specified number of bins.
     * 
     * @param data Data values
     * @param bins Number of bins
     * @param label Series label
     */
    public Histogram(double[] data, int bins, String label) {
        this.dataSeries = new double[][]{data};
        this.labels = new String[]{label};
        this.bins = bins;
    }
    
    /**
     * Create a histogram with automatic binning (Sturges' rule).
     */
    public Histogram(double[] data, String label) {
        this(data, calculateSturgeBins(data.length), label);
    }
    
    /**
     * Add another data series for overlaid histogram.
     */
    public Histogram addSeries(double[] data, String label) {
        double[][] newData = new double[dataSeries.length + 1][];
        String[] newLabels = new String[dataSeries.length + 1];
        
        System.arraycopy(dataSeries, 0, newData, 0, dataSeries.length);
        System.arraycopy(labels, 0, newLabels, 0, labels.length);
        
        newData[dataSeries.length] = data;
        newLabels[labels.length] = label;
        
        this.dataSeries = newData;
        this.labels = newLabels;
        
        return this;
    }
    
    /**
     * Set the number of bins.
     */
    public Histogram setBins(int bins) {
        this.bins = Math.max(1, bins);
        return this;
    }
    
    /**
     * Set the range for binning.
     */
    public Histogram setRange(double min, double max) {
        this.rangeMin = min;
        this.rangeMax = max;
        return this;
    }
    
    /**
     * Normalize the histogram to show probability density.
     */
    public Histogram normalize(boolean normalize) {
        this.normalize = normalize;
        return this;
    }
    
    /**
     * Use Sturges' rule for bin count: ceil(log2(n) + 1).
     */
    private static int calculateSturgeBins(int n) {
        return (int) Math.ceil(Math.log(n) / Math.log(2) + 1);
    }
    
    /**
     * Use Scott's rule for bin count.
     */
    public Histogram useScottBins() {
        // Scott's rule: bin width = 3.5 * std / n^(1/3)
        double[] firstSeries = dataSeries[0];
        double std = calculateStd(firstSeries);
        int n = firstSeries.length;
        double range = calculateRange(firstSeries);
        double binWidth = 3.5 * std / Math.pow(n, 1.0 / 3.0);
        this.bins = (int) Math.ceil(range / binWidth);
        return this;
    }
    
    /**
     * Use Freedman-Diaconis rule for bin count.
     */
    public Histogram useFreedmanDiaconisBins() {
        double[] firstSeries = dataSeries[0];
        double iqr = calculateIQR(firstSeries);
        int n = firstSeries.length;
        double range = calculateRange(firstSeries);
        double binWidth = 2.0 * iqr / Math.pow(n, 1.0 / 3.0);
        this.bins = (int) Math.ceil(range / binWidth);
        return this;
    }
    
    @Override
    public JFreeChart toChart(PlotContext context) {
        // Create dataset
        HistogramDataset dataset = new HistogramDataset();
        
        // Determine range
        double min = rangeMin != null ? rangeMin : findMin(dataSeries);
        double max = rangeMax != null ? rangeMax : findMax(dataSeries);
        
        // Add series
        for (int i = 0; i < dataSeries.length; i++) {
            dataset.addSeries(labels[i], dataSeries[i], bins, min, max);
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createHistogram(
            context.getTitle(),
            context.getXlabel(),
            context.getYlabel().isEmpty() ? (normalize ? "Density" : "Frequency") : context.getYlabel(),
            dataset,
            PlotOrientation.VERTICAL,
            context.isShowLegend(),
            true,
            false
        );
        
        // Apply context
        applyContext(chart, context);
        
        return chart;
    }
    
    private void applyContext(JFreeChart chart, PlotContext context) {
        chart.setBackgroundPaint(context.getBackgroundColor());
        chart.getTitle().setFont(context.getTitleFont());
        
        XYPlot plot = chart.getXYPlot();
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
    }
    
    private double findMin(double[][] data) {
        double min = Double.POSITIVE_INFINITY;
        for (double[] series : data) {
            for (double v : series) {
                if (v < min) min = v;
            }
        }
        return min;
    }
    
    private double findMax(double[][] data) {
        double max = Double.NEGATIVE_INFINITY;
        for (double[] series : data) {
            for (double v : series) {
                if (v > max) max = v;
            }
        }
        return max;
    }
    
    private double calculateRange(double[] data) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double v : data) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return max - min;
    }
    
    private double calculateStd(double[] data) {
        double mean = 0;
        for (double v : data) mean += v;
        mean /= data.length;
        
        double variance = 0;
        for (double v : data) {
            double diff = v - mean;
            variance += diff * diff;
        }
        variance /= data.length;
        
        return Math.sqrt(variance);
    }
    
    private double calculateIQR(double[] data) {
        double[] sorted = data.clone();
        java.util.Arrays.sort(sorted);
        
        int n = sorted.length;
        int q1Index = n / 4;
        int q3Index = 3 * n / 4;
        
        return sorted[q3Index] - sorted[q1Index];
    }
}
