package com.user.nn.utils.visualization.exporters;

import com.user.nn.utils.visualization.Plot;
import com.user.nn.utils.visualization.PlotContext;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;

import java.io.*;
import java.util.Base64;

/**
 * Export plots to interactive HTML with embedded charts.
 * The resulting HTML file is standalone and can be opened in any browser.
 * 
 * Example:
 * <pre>
 * LinePlot plot = new LinePlot(x, y, "Data");
 * HTMLExporter.saveHTML(plot, new PlotContext().title("My Plot"), "output.html");
 * </pre>
 */
public class HTMLExporter {
    
    /**
     * Save plot as standalone HTML file.
     * 
     * @param plot The plot to save
     * @param context Plot configuration
     * @param filePath Output file path
     * @throws IOException If file cannot be written
     */
    public static void saveHTML(Plot plot, PlotContext context, String filePath) throws IOException {
        saveHTML(plot, context, filePath, 800, 600);
    }
    
    /**
     * Save plot as HTML with custom dimensions.
     * 
     * @param plot The plot to save
     * @param context Plot configuration
     * @param filePath Output file path
     * @param width Width in pixels
     * @param height Height in pixels
     * @throws IOException If file cannot be written
     */
    public static void saveHTML(Plot plot, PlotContext context, String filePath, int width, int height) throws IOException {
        JFreeChart chart = plot.toChart(context);
        File file = new File(filePath);
        
        // Ensure parent directory exists
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        
        // Generate PNG and encode as base64
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ChartUtils.writeChartAsPNG(baos, chart, width, height);
        String base64Image = Base64.getEncoder().encodeToString(baos.toByteArray());
        
        // Generate HTML
        String html = generateHTML(context.getTitle(), base64Image, width, height);
        
        // Write to file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            writer.write(html);
        }
    }
    
    /**
     * Generate HTML content with embedded chart.
     */
    private static String generateHTML(String title, String base64Image, int width, int height) {
        StringBuilder html = new StringBuilder();
        
        html.append("<!DOCTYPE html>\n");
        html.append("<html lang=\"en\">\n");
        html.append("<head>\n");
        html.append("    <meta charset=\"UTF-8\">\n");
        html.append("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.append("    <title>").append(escapeHtml(title)).append("</title>\n");
        html.append("    <style>\n");
        html.append("        body {\n");
        html.append("            font-family: Arial, sans-serif;\n");
        html.append("            margin: 20px;\n");
        html.append("            background-color: #f5f5f5;\n");
        html.append("        }\n");
        html.append("        .container {\n");
        html.append("            max-width: 1200px;\n");
        html.append("            margin: 0 auto;\n");
        html.append("            background-color: white;\n");
        html.append("            padding: 20px;\n");
        html.append("            box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n");
        html.append("        }\n");
        html.append("        h1 {\n");
        html.append("            color: #333;\n");
        html.append("            margin-bottom: 20px;\n");
        html.append("        }\n");
        html.append("        .chart-container {\n");
        html.append("            text-align: center;\n");
        html.append("            margin: 20px 0;\n");
        html.append("        }\n");
        html.append("        img {\n");
        html.append("            max-width: 100%;\n");
        html.append("            height: auto;\n");
        html.append("            border: 1px solid #ddd;\n");
        html.append("        }\n");
        html.append("        .info {\n");
        html.append("            color: #666;\n");
        html.append("            font-size: 12px;\n");
        html.append("            margin-top: 20px;\n");
        html.append("            text-align: center;\n");
        html.append("        }\n");
        html.append("    </style>\n");
        html.append("</head>\n");
        html.append("<body>\n");
        html.append("    <div class=\"container\">\n");
        
        if (!title.isEmpty()) {
            html.append("        <h1>").append(escapeHtml(title)).append("</h1>\n");
        }
        
        html.append("        <div class=\"chart-container\">\n");
        html.append("            <img src=\"data:image/png;base64,").append(base64Image).append("\" ");
        html.append("alt=\"").append(escapeHtml(title)).append("\" ");
        html.append("width=\"").append(width).append("\" ");
        html.append("height=\"").append(height).append("\" />\n");
        html.append("        </div>\n");
        html.append("        <div class=\"info\">\n");
        html.append("            Generated by ML Framework Visualization<br>\n");
        html.append("            ").append(java.time.LocalDateTime.now().toString()).append("\n");
        html.append("        </div>\n");
        html.append("    </div>\n");
        html.append("</body>\n");
        html.append("</html>\n");
        
        return html.toString();
    }
    
    /**
     * Escape HTML special characters.
     */
    private static String escapeHtml(String text) {
        if (text == null) return "";
        return text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace("\"", "&quot;")
                   .replace("'", "&#39;");
    }
}
