import os
import glob
import re

files = glob.glob("src/com/user/nn/examples/Train*.java")

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # find the last "}" which closes the class
    last_brace_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if lines[i].strip() == "}":
            last_brace_idx = i
            break
            
    # find the second to last "}" which closes main
    second_last_brace_idx = -1
    for i in range(last_brace_idx-1, -1, -1):
        if lines[i].strip() == "}":
            second_last_brace_idx = i
            break
            
    if second_last_brace_idx != -1:
        # inject System.exit(0); before the second to last "}"
        lines.insert(second_last_brace_idx, "        System.exit(0);\n")
        with open(f, "w", encoding="utf-8") as file:
            file.writelines(lines)
        print(f"Added System.exit(0) to {f}")
    else:
        print(f"Could not process {f}")
