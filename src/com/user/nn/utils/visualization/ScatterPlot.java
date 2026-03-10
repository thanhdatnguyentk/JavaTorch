package com.user.nn.utils.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.awt.geom.Ellipse2D;

/**
 * Scatter plot implementation.
 * Supports color mapping, size mapping, and transparency.
 * 
 * Example:
 * <pre>
 * double[] x = {1, 2, 3, 4, 5};
 * double[] y = {2, 4, 1, 5, 3};
 * 
 * ScatterPlot plot = new ScatterPlot(x, y, "Data");
 * plot.setColors(new double[]{0, 1, 0.5, 0.8, 0.3}); // Color by value
 * plot.setSizes(new double[]{5, 10, 8, 12, 6});      // Size by value
 * </pre>
 */
public class ScatterPlot implements Plot {
    
	private double[] x;
	private double[] y;
	private String label;
	private double[] colorValues = null;
	private double[] sizeValues = null;
	private double alpha = 1.0;
	private Color baseColor = new Color(31, 119, 180);
	private double baseSize = 6.0;
    
	public ScatterPlot(double[] x, double[] y, String label) {
		if (x.length != y.length) {
			throw new IllegalArgumentException("x and y must have the same length");
		}
		this.x = x;
		this.y = y;
		this.label = label;
	}
    
	public ScatterPlot setColors(double[] values) {
		if (values.length != x.length) {
			throw new IllegalArgumentException("Color values length must match data length");
		}
		this.colorValues = values;
		return this;
	}
    
	public ScatterPlot setSizes(double[] values) {
		if (values.length != x.length) {
			throw new IllegalArgumentException("Size values length must match data length");
		}
		this.sizeValues = values;
		return this;
	}
    
	public ScatterPlot setAlpha(double alpha) {
		this.alpha = Math.max(0.0, Math.min(1.0, alpha));
		return this;
	}
    
	public ScatterPlot setBaseColor(Color color) {
		this.baseColor = color;
		return this;
	}
    
	public ScatterPlot setBaseSize(double size) {
		this.baseSize = size;
		return this;
	}
    
	@Override
	public JFreeChart toChart(PlotContext context) {
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries series = new XYSeries(label);
        
		for (int i = 0; i < x.length; i++) {
			series.add(x[i], y[i]);
		}
		dataset.addSeries(series);
        
		JFreeChart chart = ChartFactory.createScatterPlot(
			context.getTitle(), context.getXlabel(), context.getYlabel(),
			dataset, PlotOrientation.VERTICAL, context.isShowLegend(), true, false
		);
        
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
        
		if (context.getXmin() != null && context.getXmax() != null) {
			plot.getDomainAxis().setRange(context.getXmin(), context.getXmax());
		}
		if (context.getYmin() != null && context.getYmax() != null) {
			plot.getRangeAxis().setRange(context.getYmin(), context.getYmax());
		}
        
		chart.setAntiAlias(context.isAntiAlias());
        
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(false, true);
		Color finalColor = applyAlpha(baseColor, alpha);
		renderer.setSeriesPaint(0, finalColor);
        
		double shapeSize = baseSize;
		Shape shape = new Ellipse2D.Double(-shapeSize/2, -shapeSize/2, shapeSize, shapeSize);
		renderer.setSeriesShape(0, shape);
		renderer.setSeriesShapesVisible(0, true);
		renderer.setSeriesLinesVisible(0, false);
        
		plot.setRenderer(renderer);
	}
    
	private Color applyAlpha(Color color, double alpha) {
		int alphaInt = (int) (alpha * 255);
		return new Color(color.getRed(), color.getGreen(), color.getBlue(), alphaInt);
	}
}
