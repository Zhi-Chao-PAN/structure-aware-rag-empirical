
import os

html_file = r"c:\Users\22304\Desktop\structure-aware-rag-study\mobile_injection_fix.txt"

# Direct links extracted from user's earlier message
url_before = "https://s41.ax1x.com/2026/01/31/pZhRRdx.png"
url_after = "https://s41.ax1x.com/2026/01/31/pZhRWo6.jpg"

new_block = f'''<div class="case-compare">
                    <div class="case-img-box">
                        <img class="case-img" src="{url_before}" alt="术前">
                        <div class="case-label">术前</div>
                    </div>
                    <div class="case-arrow">➤</div>
                    <div class="case-img-box">
                        <img class="case-img" src="{url_after}" alt="术后">
                        <div class="case-label" style="background:rgba(2, 132, 199, 0.8)">术后</div>
                    </div>
                </div>'''

try:
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    start_marker = '<!-- 案例1: 种牙 -->'
    idx_start = content.find(start_marker)
    if idx_start == -1: raise Exception("Marker not found")
        
    idx_div = content.find('<div class="case-compare">', idx_start)
    if idx_div == -1: raise Exception("Div not found")
        
    depth = 0
    idx_end = -1
    for i in range(idx_div, len(content)):
        if content[i:].startswith('<div'): depth += 1
        elif content[i:].startswith('</div>'): depth -= 1
        if depth == 0:
            idx_end = i + 6
            break
            
    if idx_end == -1: raise Exception("Closing div not found")
    
    new_content = content[:idx_div] + new_block + content[idx_end:]
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Fixed URLs to direct links.")

except Exception as e:
    print(f"Error: {e}")
