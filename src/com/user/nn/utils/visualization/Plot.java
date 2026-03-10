package com.user.nn.utils.visualization;

import org.jfree.chart.JFreeChart;

/**
 * Base interface for all plot types.
 * Each plot can be configured with a PlotContext and rendered to a JFreeChart.
 */
public interface Plot {
    
    /**
     * Create a JFreeChart from this plot.
     * @param context The plot configuration context
     * @return JFreeChart object ready for rendering
     */
    JFreeChart toChart(PlotContext context);
    
    /**
     * Create a JFreeChart with default context.
     * @return JFreeChart object ready for rendering
     */
    default JFreeChart toChart() {
        return toChart(new PlotContext());
    }
}
