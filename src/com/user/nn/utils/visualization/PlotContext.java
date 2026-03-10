package com.user.nn.utils.visualization;

import java.awt.Color;
import java.awt.Font;

/**
 * Configuration context for plots.
 * Uses builder pattern for fluent API similar to matplotlib.
 * 
 * Example:
 * <pre>
 * PlotContext ctx = new PlotContext()
 *     .title("Training Loss")
 *     .xlabel("Epoch")
 *     .ylabel("Loss")
 *     .grid(true)
 *     .legend(true);
 * </pre>
 */
public class PlotContext {
    
    // Title
    private String title = "";
    private Font titleFont = new Font("SansSerif", Font.BOLD, 18);
    
    // Axes
    private String xlabel = "";
    private String ylabel = "";
    private Font axisLabelFont = new Font("SansSerif", Font.PLAIN, 14);
    private Font tickLabelFont = new Font("SansSerif", Font.PLAIN, 12);
    
    // Legend
    private boolean showLegend = true;
    private String legendPosition = "right"; // "right", "bottom", "top", "left"
    
    // Grid
    private boolean showGrid = true;
    private boolean showDomainGrid = true;
    private boolean showRangeGrid = true;
    
    // Size
    private int width = 800;
    private int height = 600;
    
    // Axis limits
    private Double xmin = null;
    private Double xmax = null;
    private Double ymin = null;
    private Double ymax = null;
    
    // Background
    private Color backgroundColor = Color.WHITE;
    private Color plotBackgroundColor = Color.WHITE;
    
    // Anti-aliasing
    private boolean antiAlias = true;
    
    public PlotContext() {
    }
    
    // Builder methods
    
    public PlotContext title(String title) {
        this.title = title;
        return this;
    }
    
    public PlotContext xlabel(String xlabel) {
        this.xlabel = xlabel;
        return this;
    }
    
    public PlotContext ylabel(String ylabel) {
        this.ylabel = ylabel;
        return this;
    }
    
    public PlotContext legend(boolean show) {
        this.showLegend = show;
        return this;
    }
    
    public PlotContext legendPosition(String position) {
        this.legendPosition = position;
        return this;
    }
    
    public PlotContext grid(boolean show) {
        this.showGrid = show;
        this.showDomainGrid = show;
        this.showRangeGrid = show;
        return this;
    }
    
    public PlotContext domainGrid(boolean show) {
        this.showDomainGrid = show;
        return this;
    }
    
    public PlotContext rangeGrid(boolean show) {
        this.showRangeGrid = show;
        return this;
    }
    
    public PlotContext size(int width, int height) {
        this.width = width;
        this.height = height;
        return this;
    }
    
    public PlotContext xlim(double min, double max) {
        this.xmin = min;
        this.xmax = max;
        return this;
    }
    
    public PlotContext ylim(double min, double max) {
        this.ymin = min;
        this.ymax = max;
        return this;
    }
    
    public PlotContext backgroundColor(Color color) {
        this.backgroundColor = color;
        return this;
    }
    
    public PlotContext plotBackgroundColor(Color color) {
        this.plotBackgroundColor = color;
        return this;
    }
    
    public PlotContext antiAlias(boolean enable) {
        this.antiAlias = enable;
        return this;
    }
    
    public PlotContext titleFont(Font font) {
        this.titleFont = font;
        return this;
    }
    
    public PlotContext axisLabelFont(Font font) {
        this.axisLabelFont = font;
        return this;
    }
    
    public PlotContext tickLabelFont(Font font) {
        this.tickLabelFont = font;
        return this;
    }
    
    // Getters
    
    public String getTitle() {
        return title;
    }
    
    public Font getTitleFont() {
        return titleFont;
    }
    
    public String getXlabel() {
        return xlabel;
    }
    
    public String getYlabel() {
        return ylabel;
    }
    
    public Font getAxisLabelFont() {
        return axisLabelFont;
    }
    
    public Font getTickLabelFont() {
        return tickLabelFont;
    }
    
    public boolean isShowLegend() {
        return showLegend;
    }
    
    public String getLegendPosition() {
        return legendPosition;
    }
    
    public boolean isShowGrid() {
        return showGrid;
    }
    
    public boolean isShowDomainGrid() {
        return showDomainGrid;
    }
    
    public boolean isShowRangeGrid() {
        return showRangeGrid;
    }
    
    public int getWidth() {
        return width;
    }
    
    public int getHeight() {
        return height;
    }
    
    public Double getXmin() {
        return xmin;
    }
    
    public Double getXmax() {
        return xmax;
    }
    
    public Double getYmin() {
        return ymin;
    }
    
    public Double getYmax() {
        return ymax;
    }
    
    public Color getBackgroundColor() {
        return backgroundColor;
    }
    
    public Color getPlotBackgroundColor() {
        return plotBackgroundColor;
    }
    
    public boolean isAntiAlias() {
        return antiAlias;
    }
}
