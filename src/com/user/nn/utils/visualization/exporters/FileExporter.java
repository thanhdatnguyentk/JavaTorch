package com.user.nn.utils.visualization.exporters;

import com.user.nn.utils.visualization.Plot;
import com.user.nn.utils.visualization.PlotContext;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.graphics2d.svg.SVGGraphics2D;
import org.jfree.graphics2d.svg.SVGUtils;

import java.awt.geom.Rectangle2D;
import java.io.File;
import java.io.IOException;

/**
 * Export plots to PNG and SVG file formats.
 * 
 * Example:
 * <pre>
 * LinePlot plot = new LinePlot(x, y, "Data");
 * PlotContext ctx = new PlotContext().title("My Plot");
 * 
 * FileExporter.savePNG(plot, ctx, "output.png", 800, 600);
 * FileExporter.saveSVG(plot, ctx, "output.svg", 800, 600);
 * </pre>
 */
public class FileExporter {
    
    /**
     * Save plot as PNG file.
     * 
     * @param plot The plot to save
     * @param context Plot configuration
     * @param filePath Output file path
     * @param width Width in pixels
     * @param height Height in pixels
     * @throws IOException If file cannot be written
     */
    public static void savePNG(Plot plot, PlotContext context, String filePath, int width, int height) throws IOException {
        JFreeChart chart = plot.toChart(context);
        File file = new File(filePath);
        
        // Ensure parent directory exists
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        
        ChartUtils.saveChartAsPNG(file, chart, width, height);
    }
    
    /**
     * Save plot as PNG with default context.
     */
    public static void savePNG(Plot plot, String filePath, int width, int height) throws IOException {
        savePNG(plot, new PlotContext(), filePath, width, height);
    }
    
    /**
     * Save plot as PNG with default size (800x600).
     */
    public static void savePNG(Plot plot, PlotContext context, String filePath) throws IOException {
        savePNG(plot, context, filePath, 800, 600);
    }
    
    /**
     * Save plot as SVG file (vector graphics).
     * 
     * @param plot The plot to save
     * @param context Plot configuration
     * @param filePath Output file path
     * @param width Width in pixels
     * @param height Height in pixels
     * @throws IOException If file cannot be written
     */
    public static void saveSVG(Plot plot, PlotContext context, String filePath, int width, int height) throws IOException {
        JFreeChart chart = plot.toChart(context);
        File file = new File(filePath);
        
        // Ensure parent directory exists
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        
        // Create SVG graphics
        SVGGraphics2D svg2d = new SVGGraphics2D(width, height);
        Rectangle2D drawArea = new Rectangle2D.Double(0, 0, width, height);
        chart.draw(svg2d, drawArea);
        
        // Write to file
        SVGUtils.writeToSVG(file, svg2d.getSVGElement());
    }
    
    /**
     * Save plot as SVG with default context.
     */
    public static void saveSVG(Plot plot, String filePath, int width, int height) throws IOException {
        saveSVG(plot, new PlotContext(), filePath, width, height);
    }
    
    /**
     * Save plot as SVG with default size (800x600).
     */
    public static void saveSVG(Plot plot, PlotContext context, String filePath) throws IOException {
        saveSVG(plot, context, filePath, 800, 600);
    }
    
    /**
     * Save plot as high-resolution PNG for publications (300 DPI).
     * 
     * @param plot The plot to save
     * @param context Plot configuration
     * @param filePath Output file path
     * @param widthInches Width in inches
     * @param heightInches Height in inches
     * @throws IOException If file cannot be written
     */
    public static void saveHighResPNG(Plot plot, PlotContext context, String filePath, 
                                     double widthInches, double heightInches) throws IOException {
        int dpi = 300;
        int widthPixels = (int) (widthInches * dpi);
        int heightPixels = (int) (heightInches * dpi);
        savePNG(plot, context, filePath, widthPixels, heightPixels);
    }
    
    /**
     * Save both PNG and SVG versions.
     */
    public static void saveBoth(Plot plot, PlotContext context, String baseFilePath, 
                               int width, int height) throws IOException {
        String pngPath = baseFilePath.replaceAll("\\.(png|svg)$", "") + ".png";
        String svgPath = baseFilePath.replaceAll("\\.(png|svg)$", "") + ".svg";
        
        savePNG(plot, context, pngPath, width, height);
        saveSVG(plot, context, svgPath, width, height);
    }
}
