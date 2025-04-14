from optimum.exporters.onnx import OnnxConfig
from optimum.onnxruntime import ORTModelForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
from PIL import Image
import torch

# Custom ONNX configuration for BLIP
class BlipOnnxConfig(OnnxConfig):
    def __init__(self, config: BlipConfig):
        self.config = config
        self.task = "conditional-generation"  # Updated task type for BLIP
        self._num_channels = config.vision_config.num_channels
        self._image_size = config.vision_config.image_size

    @property
    def inputs(self):
        # Define expected inputs (pixel_values for images)
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        }

    @property
    def outputs(self):
        # Define expected outputs (logits for generation)
        return {
            "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"}
        }

    def generate_dummy_inputs(self, preprocessor, **kwargs):
        # Generate dummy inputs for export
        image = Image.new("RGB", (self._image_size, self._image_size), color="gray")
        inputs = preprocessor(images=image, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"]}

    @property
    def default_onnx_opset(self):
        return 12  # Use opset 12 for compatibility

# Model ID and export path
model_id = "Salesforce/blip-image-captioning-base"
onnx_path = "blip_captioning_optimum_onnx"

# Load the original PyTorch model and processor
print(f"Loading {model_id} for export...")
model = BlipForConditionalGeneration.from_pretrained(model_id)
processor = BlipProcessor.from_pretrained(model_id)

# Create custom ONNX config
onnx_config = BlipOnnxConfig(model.config)

# Export the model using Optimum
print(f"Exporting {model_id} to ONNX using Optimum...")
try:
    ort_model = ORTModelForConditionalGeneration.from_pretrained(
        model_id,
        export=True,
        onnx_config=onnx_config,
        use_io_binding=False  # Disable IO binding for compatibility
    )
    ort_model.save_pretrained(onnx_path)
    processor.save_pretrained(onnx_path)
    print(f"Model and processor saved to {onnx_path}")
except Exception as e:
    print(f"Export failed: {e}")
    raise

# --- Test the exported model ---
print("\nLoading exported model for inference test...")
model_loaded = ORTModelForConditionalGeneration.from_pretrained(onnx_path)
processor_loaded = BlipProcessor.from_pretrained(onnx_path)

# Prepare image
try:
    image = Image.open("sample.jpg").convert("RGB")
except FileNotFoundError:
    print("sample.jpg not found. Using dummy red image.")
    image = Image.new("RGB", (384, 384), color="red")

print("Processing image and generating caption...")
inputs = processor_loaded(images=image, return_tensors="pt")

# Generate caption using the ONNX model
generated_ids = model_loaded.generate(**inputs, max_length=50)
caption = processor_loaded.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated Caption (using ONNX): >>> {caption} <<<")