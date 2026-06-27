from PIL import Image
import re
import base64
import io
import os

# ensure workspace imports resolve
import app

img_path = os.path.join(os.path.dirname(__file__), "sample_crop.jpg")
image = Image.open(img_path).convert("RGB")

# Call the pipeline: args match Gradio inputs in order
# process_image(input_image, show_category, show_subcategory, llava_prompt, subcategories_text=None)
annotated_html, summary_html = app.process_image(image, "coarse", False, None, None)

# Extract base64 PNG from annotated_html
m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", annotated_html)
if m:
    b64 = m.group(1)
    img_bytes = base64.b64decode(b64)
    with open("annotated_output.png", "wb") as f:
        f.write(img_bytes)
    print("Annotated image written to annotated_output.png")
else:
    print("No image found in annotated_html")

# Print synthesized summary HTML to stdout
print("\n--- Synthesized Summary (HTML) ---\n")
print(summary_html)
