from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Prepare a dummy input (image) for export
# Use a sample image or create a dummy tensor matching BLIP's expected input
try:
    image = Image.open(r"C:\Users\voutl\OneDrive\Pictures\ai.png").convert("RGB")  # Replace with your image path
except FileNotFoundError:
    print("ai.png not found. Using dummy input.")
    image = Image.new("RGB", (384, 384), color="gray")  # BLIP expects 384x384

inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # Shape: [1, 3, 384, 384]

# Export to ONNX
output_path = "blip_captioning.onnx"
torch.onnx.export(
    model,
    pixel_values,  # Input tensor
    output_path,   # Output file
    input_names=["pixel_values"],  # Name of input
    output_names=["logits"],       # Name of output
    dynamic_axes={
        "pixel_values": {0: "batch_size"},  # Dynamic batch size
        "logits": {0: "batch_size", 1: "sequence_length"}  # Dynamic batch and sequence
    },
    opset_version=12,  # ONNX opset version (12 is widely supported)
    do_constant_folding=True  # Optimize by folding constants
)

print(f"BLIP model exported to {output_path}")

# Optional: Simplify the ONNX model for better performance
try:
    import onnx
    import onnxsim

    model_onnx = onnx.load(output_path)
    model_simplified, check = onnxsim.simplify(model_onnx)
    if check:
        simplified_path = "blip_captioning_simplified.onnx"
        onnx.save(model_simplified, simplified_path)
        print(f"Simplified ONNX model saved to {simplified_path}")
    else:
        print("Simplification failed, using original model.")
except ImportError:
    print("onnx-simplifier not installed. Skipping simplification. Install with: pip install onnx-simplifier")
except Exception as e:
    print(f"Error during simplification: {e}")