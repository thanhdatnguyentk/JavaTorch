import os, glob, re

path = "./src/com/user/nn/examples/"
for root, dirs, files in os.walk(path):
    for f in files:
        if f.startswith("Train") and f.endswith(".java") and "ResNet" not in f:
            filepath = os.path.join(root, f)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
            
            # Use DOTALL to catch multi-line method calls
            content = re.sub(
                r'((\s*)System\.out\.printf\("[^"]*?Epoch.*?%n"[^;]*?\);)',
                r'\1\2try {\n' +
                r'\2    Map<String, Float> metrics = new HashMap<>();\n' +
                r'\2    try { metrics.put("avg_loss", avgLoss); } catch (Exception e) {}\n' +
                r'\2    try { metrics.put("total_loss", (float)totalLoss); } catch (Exception e) {}\n' +
                r'\2    try { metrics.put("epoch_loss", epochLoss); } catch (Exception e) {}\n' +
                r'\2    try { metrics.put("train_acc", trainAcc); } catch (Exception e) {}\n' +
                r'\2    try { metrics.put("test_acc", testAcc); } catch (Exception e) {}\n' +
                r'\2    history.record(epoch + 1, metrics);\n' +
                r'\2    dashboard.broadcastMetrics(epoch + 1, metrics);\n' +
                r'\2} catch (Exception dashEx) {}\n',
                content, flags=re.DOTALL
            )

            with open(filepath, "w", encoding="utf-8") as file:
                file.write(content)

print("Done patching.")
