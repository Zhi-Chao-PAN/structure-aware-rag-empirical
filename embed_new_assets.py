
import base64
import os
from PIL import Image
import io
import re

# Paths
base_dir = r"c:\Users\22304\Desktop\structure-aware-rag-study"
html_file = os.path.join(base_dir, "mobile_injection_fix.txt")
cbct_path = os.path.join(base_dir, "cbct_new.jpg")
chairs_wide_path = os.path.join(base_dir, "chairs_wide.jpg")
chairs_close_path = os.path.join(base_dir, "chairs_close.jpg")

# Max B64 char limit ~ 25KB
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
                w, h = img.size
                if w > width:
                    new_h = int(h * (width / w))
                    resized_img = img.resize((width, new_h), Image.Resampling.LANCZOS)
                else:
                    resized_img = img.copy()
                
                buffer = io.BytesIO()
                resized_img.save(buffer, format="JPEG", quality=quality)
                data = buffer.getvalue()
                b64_str = base64.b64encode(data).decode('utf-8')
                
                # print(f"Trying width={width}, quality={quality} -> Len: {len(b64_str)}")
                
                if len(b64_str) < MAX_B64_LEN:
                    return f"data:image/jpeg;base64,{b64_str}"
                
                width = int(width * 0.8)
                quality = max(20, quality - 10)
                
                if width < 100:
                    return f"data:image/jpeg;base64,{b64_str}" 
                    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

try:
    print("Optimizing new assets...")
    b64_cbct = optimize_to_base64_aggressive(cbct_path)
    b64_wide = optimize_to_base64_aggressive(chairs_wide_path)
    b64_close = optimize_to_base64_aggressive(chairs_close_path)
    
    if not (b64_cbct and b64_wide and b64_close):
        print("Failed to optimize some images.")
        exit(1)

    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Replace CBCT Image
    # Look for the card with "口腔CBCT"
    # Structure:
    # <div class="equip-card">
    #   <img class="equip-img" src="...">
    #   <div class="equip-info">
    #       <div class="equip-name">口腔CBCT</div>
    
    # We'll use a specific unique string to find the location
    target_cbct_name = '<div class="equip-name">口腔CBCT</div>'
    idx_cbct_name = content.find(target_cbct_name)
    if idx_cbct_name != -1:
        # Find the preceding img tag start
        idx_img_start = content.rfind('<img class="equip-img"', 0, idx_cbct_name)
        if idx_img_start != -1:
            # Find the src attribute within this tag
            idx_src_start = content.find('src="', idx_img_start)
            idx_src_end = content.find('"', idx_src_start + 5)
            if idx_src_start != -1 and idx_src_end != -1:
                print("Replacing CBCT image...")
                content = content[:idx_src_start+5] + b64_cbct + content[idx_src_end:]

    # 2. Replace Private Treatment Room Image (Wide)
    target_env1 = '<div class="env-title">私密独立诊室</div>'
    idx_env1 = content.find(target_env1)
    if idx_env1 != -1:
        # This is env-card[1] (based on snippet it looks like a second card)
        # Scan backwards for <div class="env-card"> then forward to img
        # Better: scan backwards for img tag
        idx_img_start = content.rfind('<img class="env-img"', 0, idx_env1)
        if idx_img_start != -1:
             idx_src_start = content.find('src="', idx_img_start)
             idx_src_end = content.find('"', idx_src_start + 5)
             if idx_src_start != -1 and idx_src_end != -1:
                 print("Replacing Env Wide image...")
                 content = content[:idx_src_start+5] + b64_wide + content[idx_src_end:]

    # 3. Replace Reception -> Comfort Area (Close)
    target_env2 = '<div class="env-title">温馨候诊区</div>'
    idx_env2 = content.find(target_env2)
    if idx_env2 != -1:
        # Replace Title
        print("Updating Title...")
        content = content.replace('温馨候诊区', '舒适诊疗区')
        
        # Re-find index since content changed length? No, replace creates new string but indexes shift.
        # Actually replace is global? Let's just create new content carefully or use the found index.
        # Let's start fresh search on new content variable if needed, but simple string replace works for unique title.
        
        # Now find the Image for this section.
        # It was just renamed to 舒适诊疗区
        idx_new_title = content.find('舒适诊疗区')
        idx_img_start = content.rfind('<img class="env-img"', 0, idx_new_title)
        if idx_img_start != -1:
             idx_src_start = content.find('src="', idx_img_start)
             idx_src_end = content.find('"', idx_src_start + 5)
             if idx_src_start != -1 and idx_src_end != -1:
                 print("Replacing Env Close image...")
                 content = content[:idx_src_start+5] + b64_close + content[idx_src_end:]

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Successfully embedded new assets.")

except Exception as e:
    print(f"Error: {e}")
