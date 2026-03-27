import re

def update_readme(path, lang):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if lang == 'en':
        new_bullet = "- **Real-time Web Dashboard**: Local Javalin + Vue 3 UI for live Chart.js metrics, VRAM monitoring, and interactive inference playgrounds (Image & Text)."
    else:
        new_bullet = "- **Dashboard realtime trên Web**: Local Javalin + giao diện Vue 3 để vẽ biểu đồ Chart.js trực tiếp, theo dõi VRAM và Playground API dự đoán tương tác (Hình ảnh & Văn bản)."

    if new_bullet not in content:
        # insert after "- Prediction library with"
        match = re.search(r'- Prediction library.*?$', content, re.MULTILINE)
        if match:
            pos = match.end()
            content = content[:pos] + '\n' + new_bullet + content[pos:]
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated {path}")
        else:
            print(f"Could not find insert position in {path}")

update_readme('README.md', 'en')
update_readme('README.vn.md', 'vn')
