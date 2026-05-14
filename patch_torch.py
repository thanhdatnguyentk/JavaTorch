import re

with open("d:\\My\\University\\Nam3\\HK2\\Java\\Machine_learning\\framework\\ML_framework\\src\\com\\user\\nn\\core\\Torch.java", "r") as f:
    content = f.read()

content = re.sub(r'\.toCPU\(\)', '.syncToHost()', content)

with open("d:\\My\\University\\Nam3\\HK2\\Java\\Machine_learning\\framework\\ML_framework\\src\\com\\user\\nn\\core\\Torch.java", "w") as f:
    f.write(content)

print("Patch applied to Torch.java")
