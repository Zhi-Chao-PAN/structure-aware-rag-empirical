
import os

# Paths
base_dir = r"c:\Users\22304\Desktop\structure-aware-rag-study"
html_file = os.path.join(base_dir, "mobile_injection_fix.txt")

# The text to insert
new_block = '''<div class="case-compare">
                    <div class="case-img-box">
                        <img class="case-img" src="https://s41.ax1x.com/2026/01/31/pZhRRdx.png" alt="术前">
                        <div class="case-label">术前</div>
                    </div>
                    <div class="case-arrow">➤</div>
                    <div class="case-img-box">
                        <img class="case-img" src="https://s41.ax1x.com/2026/01/31/pZhRWo6.jpg" alt="术后">
                        <div class="case-label" style="background:rgba(2, 132, 199, 0.8)">术后</div>
                    </div>
                </div>'''

try:
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # We will search for the specific case study 1 block. 
    start_marker = '<!-- 案例1: 种牙 -->'
    idx_start = content.find(start_marker)
    
    if idx_start == -1:
        print("Could not find Case 1 marker")
        exit(1)
        
    # Find the case-compare div after the marker
    idx_div = content.find('<div class="case-compare">', idx_start)
    if idx_div == -1:
        print("Could not find case-compare div")
        exit(1)
        
    # Find the closing div for case-compare.
    # We match nested divs logic again to be safe
    depth = 0
    idx_end = -1
    for i in range(idx_div, len(content)):
        if content[i:].startswith('<div'):
            depth += 1
        elif content[i:].startswith('</div>'):
            depth -= 1
            if depth == 0:
                idx_end = i + 6 # include </div>
                break
    
    if idx_end == -1:
        print("Could not parse div structure")
        exit(1)
        
    print(f"Replacing section from index {idx_div} to {idx_end}")
    
    new_content = content[:idx_div] + new_block + content[idx_end:]
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Successfully replaced images with external URLs.")

except Exception as e:
    print(f"Error: {e}")
