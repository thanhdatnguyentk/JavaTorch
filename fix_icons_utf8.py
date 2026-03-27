import re
with open('core/src/main/resources/public/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('updateCharts();', 'updateCharts();\n                            lucide.createIcons();')

with open('core/src/main/resources/public/index.html', 'w', encoding='utf-8') as f:
    f.write(text)
