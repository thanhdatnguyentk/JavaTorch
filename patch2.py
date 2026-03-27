# -*- coding: utf-8 -*-
import os, re

path = "./src/com/user/nn/examples/"

def parse_and_inject(file_path):
    print("Processing", file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'\s+for\s*\(\s*int\s+epoch\s*=\s*[01]\s*;\s*epoch\s*[<=]\s*[A-Za-z0-9_]+\s*;\s*epoch\+\+\s*\)\s*\{', content)
    if not match:
        return
    
    start_idx = match.end()
    brace_count = 1
    end_idx = -1
    for i in range(start_idx, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
                
    if end_idx == -1: return
    
    if "dashboard.broadcastMetrics" in content: return

    loop_body = content[start_idx:end_idx]
    
    vars_to_add = []
    
    # Check what variables we have safely declared at the end of the loop
    # or what is used in the final print statement
    if "float avgLoss =" in loop_body or "avgLoss = " in loop_body:
        vars_to_add.append('metrics.put("loss", avgLoss);')
    elif "float epochLoss =" in loop_body or "epochLoss = " in loop_body:
        vars_to_add.append('metrics.put("loss", epochLoss);')
        
    if "float trainAcc =" in loop_body or "trainAcc =" in loop_body:
        vars_to_add.append('metrics.put("train_acc", trainAcc);')
        
    if "float testAcc =" in loop_body or "testAcc =" in loop_body:
        vars_to_add.append('metrics.put("test_acc", testAcc);')
        
    if "float test_acc =" in loop_body:
        vars_to_add.append('metrics.put("test_acc", test_acc);')
        
    if not vars_to_add:
        # Generic fallback
        vars_to_add.append('metrics.put("epoch", (float)epoch);')
        
    puts = "\n                ".join(vars_to_add)
    
    metrics_injection = f"""
            try {{
                Map<String, Float> metrics = new HashMap<>();
                {puts}
                history.record(epoch + 1, metrics);
                dashboard.broadcastMetrics(epoch + 1, metrics);
            }} catch (Exception dashEx) {{}}
"""
    new_content = content[:end_idx] + metrics_injection + content[end_idx:]
    
    # Add shutdown
    new_content = re.sub(r'(System\.out\.println\("\\nTraining Complete!"\);)', r'\1\n        try { dashboard.exportDashboardData("dashboard_final.json"); } catch(Exception e) {}', new_content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

for root, dirs, files in os.walk(path):
    for f in files:
        if f.startswith("Train") and f.endswith(".java") and "ResNetCifar10" not in f:
            parse_and_inject(os.path.join(root, f))

print("Done phase 2.")
