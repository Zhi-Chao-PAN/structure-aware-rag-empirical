
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

# Target Max Base64 Length ~ 30,000 chars (~22KB)
MAX_B64_LEN = 35000 

def optimize_to_base64_aggressive(image_path, start_width=300):
    try:
        if not os.path.exists(image_path):
            print(f"Error: File not found {image_path}")
            return None

        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            width = start_width
            quality = 60
            
            while True:
                # Resize
                w, h = img.size
                if w > width:
                    new_h = int(h * (width / w))
                    resized_img = img.resize((width, new_h), Image.Resampling.LANCZOS)
                else:
                    resized_img = img.copy()
                
                # Save to buffer
                buffer = io.BytesIO()
                resized_img.save(buffer, format="JPEG", quality=quality)
                data = buffer.getvalue()
                b64_str = base64.b64encode(data).decode('utf-8')
                
                print(f"Trying width={width}, quality={quality} -> Len: {len(b64_str)}")
                
                if len(b64_str) < MAX_B64_LEN:
                    return f"data:image/jpeg;base64,{b64_str}"
                
                # Reduce params for next loop
                width = int(width * 0.8)
                quality = max(20, quality - 10)
                
                if width < 100:
                    print("Could not compress enough!")
                    return f"data:image/jpeg;base64,{b64_str}" # Return what we have
                    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

try:
    print("Aggressively optimizing images...")
    b64_before = optimize_to_base64_aggressive(img_before_path)
    b64_after = optimize_to_base64_aggressive(img_after_path)
    
    if not b64_before or not b64_after:
        print("Failed to optimize images.")
        exit(1)

    print(f"Final Before img b64 length: {len(b64_before)}")
    print(f"Final After img b64 length: {len(b64_after)}")

    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

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

    # Robust replacement logic
    start_marker = '<!-- 案例1: 种牙 -->'
    idx_start = content.find(start_marker)
    
    if idx_start == -1:
        print("Could not find Case 1 marker")
        exit(1)
        
    idx_div = content.find('<div class="case-compare">', idx_start)
    if idx_div == -1:
        print("Could not find case-compare div")
        exit(1)
        
    # Find closing div
    depth = 0
    idx_end = -1
    for i in range(idx_div, len(content)):
        if content[i:].startswith('<div'):
            depth += 1
        elif content[i:].startswith('</div>'):
            depth -= 1
            if depth == 0:
                idx_end = i + 6
                break
    
    if idx_end == -1:
        print("Could not parse div structure")
        exit(1)
        
    print(f"Replacing section from index {idx_div} to {idx_end}")
    new_content = content[:idx_div] + new_block + content[idx_end:]
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Successfully injected optimized base64 images.")

except Exception as e:
    print(f"Error: {e}")
