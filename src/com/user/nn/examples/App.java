package com.user.nn.examples;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.Linear;
import com.user.nn.optim.Optim;
import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.visualization.TrainingHistory;

import java.util.HashMap;
import java.util.Map;

public class App {
    
    // Modern XOR Model using high-level API
    public static class XORModel extends Module {
        public Linear fc1;
        public Linear fc2;

        public XORModel() {
            this.fc1 = new Linear(2, 4, true);
            this.fc2 = new Linear(4, 1, true);
            
            addModule("fc1", fc1);
            addModule("fc2", fc2);
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = fc1.forward(x);
            h = Torch.sigmoid(h);
            return Torch.sigmoid(fc2.forward(h));
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== JavaTorch Modern XOR Example ===");

        // 1. Data Setup (XOR)
        float[] xData = {
            0, 0,
            0, 1,
            1, 0,
            1, 1
        };
        float[] yData = { 0, 1, 1, 0 };

        Tensor X = Torch.tensor(xData, 4, 2);
        Tensor Y = Torch.tensor(yData, 4, 1);

        // 2. Model & Optimizer
        XORModel model = new XORModel();
        float lr = 0.5f;
        int epochs = SmokeTest.getEpochs(2000); // 2k is enough for XOR with modern init
        Optim.SGD optimizer = new Optim.SGD(model.parameters(), lr);

        // 3. Dashboard setup (optional visualization)
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
        dashboard.setTaskType("classification");
        dashboard.setModelInfo("Modern XOR MLP", epochs);

        // 4. Training Loop
        for (int e = 0; e < epochs; e++) {
            optimizer.zero_grad();
            
            // Forward
            Tensor output = model.forward(X);
            
            // Binary Cross Entropy or MSE
            Tensor loss = Functional.mse_loss_tensor(output, Y);
            
            // Backward
            loss.backward();
            
            // Update
            optimizer.step();

            if (e % 100 == 0 || e == epochs - 1) {
                float lossVal = loss.data[0];
                System.out.printf("Epoch %d | Loss: %.6f%n", e, lossVal);
                
                // Dashboard update
                Map<String, Float> metrics = new HashMap<>();
                metrics.put("loss", lossVal);
                dashboard.broadcastMetrics(e + 1, metrics);
            }
            
            while (dashboard.isTrainingPaused()) {
                Thread.sleep(200);
            }
        }

        // 5. Final Evaluation
        model.eval();
        Tensor finalOut = model.forward(X);
        System.out.println("\nFinal Results:");
        for (int i = 0; i < 4; i++) {
            System.out.printf("Input: [%.0f, %.0f] -> Predict: %.4f (Actual: %.0f)%n", 
                X.data[i*2], X.data[i*2+1], finalOut.data[i], Y.data[i]);
        }
        
        System.out.println("\nTraining complete. You can view progress at http://localhost:7070");
    }
}
