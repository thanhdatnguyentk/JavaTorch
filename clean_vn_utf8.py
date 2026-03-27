import re

path = 'README.vn.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('- **Dashboard realtime trÃªn Web**: Local Javalin + giao dián Vue 3 Äáƒ váº\n biáƒu Äá Chart.js trác tiáºp, theo dÃµi VRAM vÃ Playground API dá Ä\noÃn tÆÆng tÃc (HÃnh áºnh & VÄƒn báºn).', '')

# Remove any weird insertions of dashboard
content = re.sub(r'- \*\*Dashboard realtime.*?\n', '', content, flags=re.MULTILINE)

new_bullet = "- **Dashboard realtime trên Web**: Local Javalin + giao diện Vue 3 để vẽ biểu đồ Chart.js trực tiếp, theo dõi VRAM và Playground API dự đoán tương tác (Hình ảnh & Văn bản)."

match = re.search(r'- Thu vien predict voi.*?$', content, re.MULTILINE)
if match:
    pos = match.end()
    content = content[:pos] + '\n' + new_bullet + content[pos:]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
