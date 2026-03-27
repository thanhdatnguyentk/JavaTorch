import re

path = 'README.vn.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

new_bullet = "- **Dashboard realtime trên Web**: Local Javalin + giao diện Vue 3 để vẽ biểu đồ Chart.js trực tiếp, theo dõi VRAM và Playground API dự đoán tương tác (Hình ảnh & Văn bản)."

if new_bullet not in content:
    # insert after "- Thu vien predict voi"
    match = re.search(r'- Thu vien predict voi.*?$', content, re.MULTILINE)
    if match:
        pos = match.end()
        content = content[:pos] + '\n' + new_bullet + content[pos:]
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {path}")
    else:
        print(f"Could not find insert position in {path}")
