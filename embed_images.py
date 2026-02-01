
import base64
import os

# Paths
base_dir = r"c:\Users\22304\Desktop\structure-aware-rag-study"
html_file = os.path.join(base_dir, "mobile_injection_fix.txt")
img_before = os.path.join(base_dir, "implant_before.jpg")
img_after = os.path.join(base_dir, "implant_after.jpg")

# Function to get base64 string
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"

try:
    # Generate Base64 strings
    b64_before = get_base64_image(img_before)
    b64_after = get_base64_image(img_after)

    # Read HTML file
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace placeholders
    # We look for the exact strings we put in previously
    content = content.replace('src="./implant_before.jpg"', f'src="{b64_before}"')
    content = content.replace('src="./implant_after.jpg"', f'src="{b64_after}"')

    # Write back
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("Successfully embedded images as Base64.")

except Exception as e:
    print(f"Error: {e}")
