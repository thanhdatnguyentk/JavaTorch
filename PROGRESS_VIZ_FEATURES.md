# Progress Bar & Visualization Features

## Overview

Đã thêm 2 utilities quan trọng vào ML framework:

### 1. **Progress Bar** (giống tqdm của Python)
   - Package: `com.user.nn.utils.progress`
   - Features:
     - ANSI terminal progress bars với auto-refresh
     - Hiển thị metrics real-time (loss, accuracy, throughput, ETA)
     - ProgressDataLoader wrapper cho DataLoader
     - Thread-safe, hỗ trợ multi-threaded training

### 2. **Visualization** (giống matplotlib của Python)
   - Package: `com.user.nn.utils.visualization`
   - Features:
     - **Plot types**: LinePlot, ScatterPlot, Histogram, HeatmapPlot, BarChart
     - **Export formats**: PNG, SVG, HTML (interactive)
     - **Real-time viewing**: Swing window với zoom/pan
     - **ML-specific helpers**: confusion matrix, training curves, gradient flow, embeddings
     - **TrainingHistory**: Auto-track metrics qua epochs, export CSV


### 3. **Real-time Web Dashboard & Inference Playground**
   - Package: com.user.nn.utils.dashboard
   - Giao dien web hien thi huan luyen theo thoi gian thuc.
   - Theo doi **VRAM process, VRAM pool, VRAM System**.
   - **Vue 3 + Chart.js realtime (WebSocket)** cap nhat moi epoch.
   - **Playground (Inference)**: Upload hinh anh hoac nhap text de test model prediction thong qua DashboardIntegrationHelper.
   - Start 1 lan 
ew DashboardServer(7070, history).start(), giao dien web nam tai http://localhost:7070.

## Quick Start

### Progress Bar

```java
import com.user.nn.utils.progress.ProgressBar;

// Basic usage
ProgressBar bar = new ProgressBar(100, "Processing");
for (int i = 0; i < 100; i++) {
    // Do work...
    bar.update(1);
    bar.setPostfix("loss", currentLoss);
}
bar.close();

// With DataLoader
import com.user.nn.utils.progress.ProgressDataLoader;

DataLoader loader = new DataLoader(dataset, 32, true, 2);
ProgressDataLoader progLoader = new ProgressDataLoader(loader, "Training");

for (Tensor[] batch : progLoader) {
    // Training code...
    progLoader.setPostfix("loss", loss.data[0]);
}
```

### Visualization

```java
import com.user.nn.utils.visualization.*;
import com.user.nn.utils.visualization.exporters.FileExporter;

// Line plot
double[] x = {1, 2, 3, 4, 5};
double[] y = {2, 4, 3, 5, 4};
LinePlot plot = new LinePlot(x, y, "Data");

PlotContext ctx = new PlotContext()
    .title("My Plot")
    .xlabel("X Axis")
    .ylabel("Y Axis")
    .grid(true);

// Save to file
FileExporter.savePNG(plot, ctx, "output.png");
FileExporter.saveSVG(plot, ctx, "output.svg");

// Show in window
import com.user.nn.utils.visualization.viewers.PlotViewer;
PlotViewer.show(plot, ctx);
```

### Training History

```java
import com.user.nn.utils.visualization.TrainingHistory;

TrainingHistory history = new TrainingHistory();

for (int epoch = 0; epoch < epochs; epoch++) {
    // Training...
    
    Map<String, Float> metrics = new HashMap<>();
    metrics.put("train_loss", trainLoss);
    metrics.put("val_acc", valAcc);
    history.record(epoch, metrics);
}

// Plot curves
Plot plot = history.plot();
FileExporter.savePNG(plot, "training_curves.png");

// Export to CSV
history.saveCSV("history.csv");

// Get best results
float bestAcc = history.getMax("val_acc");
int bestEpoch = history.getMaxEpoch("val_acc");
```

### ML-Specific Visualizations

```java
import com.user.nn.utils.visualization.MLViz;

// Confusion matrix
int[][] cm = {{50, 2}, {3, 45}};
String[] labels = {"Class 0", "Class 1"};
HeatmapPlot cmPlot = MLViz.plotConfusionMatrix(cm, labels);

// Training curves from history
LinePlot curves = MLViz.plotTrainingCurves(history);

// Weight distribution
Tensor weights = layer.getParameter("weight").getTensor();
Histogram hist = MLViz.plotWeightDistribution(weights);

// Gradient flow
BarChart gradFlow = MLViz.plotGradientFlow(model);

// 2D embeddings
Tensor embeddings = ...; // N x 2
int[] labels = ...; // class labels
ScatterPlot scatter = MLViz.plotEmbeddings2D(embeddings, labels);
```

## Plot Types

### Line Plot
- Multiple series với colors/styles khác nhau
- Markers (circle, square, triangle, diamond)
- Line styles (solid, dashed, dotted)

### Scatter Plot
- Color mapping by value
- Size mapping by value
- Alpha/transparency control

### Histogram
- Auto binning algorithms (Sturges, Scott, Freedman-Diaconis)
- Overlaid histograms for comparison
- Normalization option

### Heatmap
- Colormaps: viridis, plasma, coolwarm, RdBu
- Custom tick labels
- Annotations in cells

### Bar Chart
- Grouped bars
- Horizontal/vertical orientation

## Files Structure

```
src/com/user/nn/utils/
├── progress/
│   ├── AnsiCodes.java          # ANSI terminal utilities
│   ├── ProgressBar.java        # Core progress bar
│   ├── ProgressDataLoader.java # DataLoader wrapper
│   └── MetricFormatter.java    # Format utilities
└── visualization/
    ├── Plot.java               # Base interface
    ├── PlotContext.java        # Configuration builder
    ├── LinePlot.java           # Line plots
    ├── ScatterPlot.java        # Scatter plots
    ├── Histogram.java          # Histograms
    ├── HeatmapPlot.java        # Heatmaps
    ├── BarChart.java           # Bar charts
    ├── TrainingHistory.java    # Training tracking
    ├── MLViz.java              # ML-specific helpers
    ├── exporters/
    │   ├── FileExporter.java   # PNG/SVG export
    │   └── HTMLExporter.java   # HTML export
    └── viewers/
        └── PlotViewer.java     # Interactive viewing
```

## Dependencies

Added to `core/build.gradle.kts`:
```kotlin
implementation("org.jfree:jfreechart:1.5.4")
implementation("org.jfree:jfreesvg:3.4")
```

## Examples

Run comprehensive demo:
```bash
# Demo all features
java -cp "core/build/classes/java/main:..." \
  com.user.nn.examples.ProgressAndVisualizationDemo
```

## Notes

- Progress bar sử dụng ANSI codes, hoạt động tốt nhất với modern terminals (Windows Terminal, Linux terminals)
- Visualization sử dụng JFreeChart backend, mature và well-documented
- All features hoàn toàn optional - không breaking changes
- Thread-safe cho multi-threaded training
- Headless mode supported cho CI/CD (PNG/SVG export vẫn hoạt động)

## Future Enhancements (Optional)

- 3D surface plots
- Animation/video export
- TensorBoard-like web dashboard
- Distributed training progress aggregation
- Custom colormaps beyond JFreeChart defaults
- LaTeX rendering in labels

---

**Status**: ✅ Implementation complete, compilation successful
**Ready for**: Production use, documentation update, unit testing
