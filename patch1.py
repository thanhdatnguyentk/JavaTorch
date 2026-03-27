# -*- coding: utf-8 -*-
import os, glob, re

path = "./src/com/user/nn/examples/"
for root, dirs, files in os.walk(path):
    for f in files:
        if f.startswith("Train") and f.endswith(".java") and "ResNetCifar10" not in f:
            filepath = os.path.join(root, f)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
            
            # IMPORTS
            imports = """
import com.user.nn.utils.dashboard.DashboardServer;
import com.user.nn.utils.dashboard.DashboardIntegrationHelper;
import com.user.nn.utils.visualization.TrainingHistory;
import java.util.HashMap;
import java.util.Map;
"""
            if "DashboardServer" not in content:
                content = re.sub(r'(package .*;)', r'\1\n' + imports, content, count=1)
            
            # SETUP Dashboard
            match_loop = re.search(r'\s+for\s*\(\s*int\s+epoch\s*=\s*[01]\s*;\s*epoch\s*[<=]\s*[A-Za-z0-9_]+\s*;\s*epoch\+\+\s*\)\s*\{', content)
            if match_loop and "DashboardServer dashboard" not in content:
                setup_str = f"""
        TrainingHistory history = new TrainingHistory();
        DashboardServer dashboard = new DashboardServer(7070, history).start();
"""
                
                predictor_setup = ""
                if "CIFAR10" in content.upper() or "Cifar10" in content:
                    predictor_setup = """
        try {
            com.user.nn.predict.ImagePredictor predictor = com.user.nn.predict.ImagePredictor.forCifar10(model);
            DashboardIntegrationHelper.setupImagePredictorHandler(dashboard, "classify_image", predictor);
        } catch(Exception e) {}
"""
                elif "MNIST" in content.upper() or "Mnist" in content:
                    predictor_setup = """
        try {
            com.user.nn.predict.ImagePredictor predictor = com.user.nn.predict.ImagePredictor.forMnist(model);
            DashboardIntegrationHelper.setupImagePredictorHandler(dashboard, "classify_image", predictor);
        } catch(Exception e) {}
"""
                elif "FASHION" in content.upper():
                    predictor_setup = """
        try {
            com.user.nn.predict.ImagePredictor predictor = com.user.nn.predict.ImagePredictor.forFashionMnist(model);
            DashboardIntegrationHelper.setupImagePredictorHandler(dashboard, "classify_image", predictor);
        } catch(Exception e) {}
"""
                elif "Sentiment" in content:
                     predictor_setup = """
        try {
            com.user.nn.predict.TextPredictor predictor = com.user.nn.predict.TextPredictor.forSentiment(model, vocab, maxLen);
            DashboardIntegrationHelper.setupTextPredictorHandler(dashboard, "sentiment", predictor);
        } catch(Exception e) {}
"""

                before_loop = content[:match_loop.start()]
                after_loop = content[match_loop.start():]
                
                content = before_loop + setup_str + predictor_setup + after_loop
            
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(content)

print("Done phase 1.")
