
import base64
import os
from PIL import Image
import io
import re

# Paths
base_dir = r"c:\Users\22304\Desktop\structure-aware-rag-study"
html_file = os.path.join(base_dir, "mobile_injection_fix.txt")
img_before_path = os.path.join(base_dir, "implant_before.jpg")
img_after_path = os.path.join(base_dir, "implant_after.jpg")

def optimize_to_base64(image_path, max_width=400):
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed (e.g. if PNG with transparency)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Resize if too big
            w, h = img.size
            if w > max_width:
                new_h = int(h * (max_width / w))
                img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
            
            # Save to buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            data = buffer.getvalue()
            
            # Encode
            b64_str = base64.b64encode(data).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_str}"
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

try:
    print("Optimizing images...")
    b64_before = optimize_to_base64(img_before_path)
    b64_after = optimize_to_base64(img_after_path)
    
    if not b64_before or not b64_after:
        print("Failed to optimize images.")
        exit(1)

    print(f"Before img b64 length: {len(b64_before)}")
    print(f"After img b64 length: {len(b64_after)}")

    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to replace the images in the specific section
    # We look for the case-compare block and replace specifically the img src attributes
    # Because the previous replacement might have left a mess, we'll reconstruct the block.
    
    target_block_start = '<div class="case-compare">'
    target_block_end = '</div>'
    
    # We will search for the specific case study 1 block. 
    # It starts with <!-- 案例1: 种牙 -->
    # Then has <div class="case-compare">...</div>
    
    # Let's find the location
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
    # It contains two case-img-box divs, so we need to be careful.
    # Structure:
    # <div class="case-compare">
    #    <div ...> <img ...> ... </div>
    #    <div class="case-arrow">...</div>
    #    <div ...> <img ...> ... </div>
    # </div>
    
    # Simplest way: just replace the whole inner HTML of this specific case-compare block if we can identify it.
    
    new_block = f'''<div class="case-compare">
                    <div class="case-img-box">
                        <img class="case-img" src="{b64_before}" alt="术前">
                        <div class="case-label">术前</div>
                    </div>
                    <div class="case-arrow">➤</div>
                    <div class="case-img-box">
                        <img class="case-img" src="{b64_after}" alt="术后">
                        <div class="case-label" style="background:rgba(2, 132, 199, 0.8)">术后</div>
                    </div>
                </div>'''

    # We need to replace the EXISTING div. 
    # We can use regex to match the div and its closing tag, allowing for nested divs inside.
    # Or since we know the structure has exactly 3 children divs (box, arrow, box), we can match until the </div> that closes case-compare.
    
    # Let's verify what's currently there. It might be huge base64 strings.
    # We will construct a regex that matches <div class="case-compare"> ... </div> 
    # but since regex is greedy, we need to limit it. 
    # Actually, let's just use Python string manipulation to be safe against huge text.
    
    # We found idx_div. Now we need to find the matching closing div.
    # Count valid <div> and </div>
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
        
    old_section = content[idx_div:idx_end]
    print(f"Replacing section of length {len(old_section)} with new section of length {len(new_block)}")
    
    new_content = content[:idx_div] + new_block + content[idx_end:]
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Successfully injected optimized base64 images.")

except Exception as e:
    print(f"Error: {e}")
