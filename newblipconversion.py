from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import BlipProcessor
from PIL import Image

model_id = "Salesforce/blip-image-captioning-base"
onnx_path = "blip_captioning_optimum_onnx" # Output directory

# Export the model using Optimum - this handles encoder/decoder export
# This might take some time
print(f"Exporting {model_id} to ONNX using Optimum...")
model = ORTModelForVision2Seq.from_pretrained(model_id, export=True)
processor = BlipProcessor.from_pretrained(model_id)

# Save the processor and the exported ONNX model files
model.save_pretrained(onnx_path)
processor.save_pretrained(onnx_path)
print(f"Model and processor saved to {onnx_path}")

# --- Example of how to use the exported Optimum model ---
print("\nLoading exported model for inference test...")
model_loaded = ORTModelForVision2Seq.from_pretrained(onnx_path)
processor_loaded = BlipProcessor.from_pretrained(onnx_path)

# Prepare image
try:
    image = Image.open("sample.jpg").convert("RGB")
except FileNotFoundError:
    print("sample.jpg not found. Using dummy red image.")
    image = Image.new('RGB', (384, 384), color = 'red')

print("Processing image and generating caption...")
inputs = processor_loaded(images=image, return_tensors="pt")

# Optimum's generate function uses the ONNX sessions
generated_ids = model_loaded.generate(**inputs)
caption = processor_loaded.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated Caption (using ONNX): >>> {caption} <<<")