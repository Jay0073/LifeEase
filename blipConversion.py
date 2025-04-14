import torch
from PIL import Image
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np # Import numpy for ORT inputs
import traceback # Import traceback

# Consistent dummy text for export and verification
DUMMY_TEXT = "a test text for export and verification"
# Define image size (important for consistency)
IMG_SIZE = 384

def get_dummy_inputs(processor: BlipProcessor):
    """Helper function to create consistent dummy inputs."""
    # Image input
    dummy_image_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # Text input
    text_inputs = processor(text=DUMMY_TEXT, return_tensors="pt", padding="max_length", max_length=20, truncation=True) # Pad/truncate for fixed size

    # Combine
    dummy_torch_inputs = {
        "pixel_values": dummy_image_tensor,
        "input_ids": text_inputs.input_ids,
        "attention_mask": text_inputs.attention_mask
    }

    # Also prepare numpy versions for ONNX Runtime verification
    dummy_ort_inputs = {
        'pixel_values': dummy_image_tensor.numpy(),
        'input_ids': text_inputs.input_ids.numpy(),
        'attention_mask': text_inputs.attention_mask.numpy()
    }
    return dummy_torch_inputs, dummy_ort_inputs


def export_blip_to_onnx(model_id: str, output_dir: str):
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load model and processor
        print("Loading BLIP model and processor...")
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        processor = BlipProcessor.from_pretrained(model_id)

        # Set model to evaluation mode
        model.eval()

        # Create consistent dummy inputs using helper function
        dummy_torch_inputs, _ = get_dummy_inputs(processor) # Only need torch inputs for export

        # Export path
        onnx_path = os.path.join(output_dir, "model.onnx")

        # Dynamic axes definition (essential for variable batch size)
        dynamic_axes = {
            'pixel_values': {0: 'batch_size'},
            'input_ids': {0: 'batch_size', 1: 'sequence_length'}, # Mark sequence as dynamic for inputs
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'}, # Mark sequence as dynamic for inputs
            'logits': {0: 'batch_size', 1: 'sequence_length'}  # Mark sequence as dynamic for outputs too
        }

        # Export the model
        print("Exporting model to ONNX...")
        # Pass inputs directly as args if model.forward expects them positionally,
        # or handle dict input carefully if needed. Let's assume positional args:
        # pixel_values, input_ids, attention_mask are common inputs. Check model.forward signature.
        # For BlipForConditionalGeneration, it often takes these + decoder_input_ids etc.
        # Exporting the whole model this way for generation is inherently problematic.
        # We'll stick to the user's original input structure for now, acknowledging its limitations.
        torch.onnx.export(
            model,
            # Pass the values from the dict as a tuple of args
            (dummy_torch_inputs['pixel_values'],
             dummy_torch_inputs['input_ids'],
             dummy_torch_inputs['attention_mask']),
            onnx_path,
            input_names=['pixel_values', 'input_ids', 'attention_mask'],
            output_names=['logits'], # Still only exports logits from single pass
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=14, # Keeping opset 14
            verbose=False
        )

        # Save the processor
        processor.save_pretrained(output_dir)
        print(f"Model exported successfully to {onnx_path}")

        return True, onnx_path

    except Exception as e:
        print(f"Export failed: {str(e)}")
        traceback.print_exc()
        return False, None

def verify_onnx_model(onnx_path: str, processor: BlipProcessor): # Pass processor
    try:
        import onnx
        import onnxruntime

        print("\nVerifying ONNX model...")
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX checker passed.")

        # Create ONNX Runtime session
        session = onnxruntime.InferenceSession(onnx_path)
        print("ONNX Runtime session created.")

        # Create consistent dummy inputs for testing using helper function
        _, dummy_ort_inputs = get_dummy_inputs(processor) # Use helper

        # Run inference
        print("Running inference with ONNX Runtime...")
        outputs = session.run(None, dummy_ort_inputs) # Use the consistent inputs
        print(f"ONNX model verification successful! Output logits shape: {outputs[0].shape}")
        return True

    except Exception as e:
        print(f"ONNX model verification failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    # Configuration
    model_id = "Salesforce/blip-image-captioning-base"
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "." # Handle interactive use
    output_dir = os.path.join(current_dir, "blip_output")

    print(f"PyTorch version: {torch.__version__}")
    try:
        import onnxruntime
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError:
        print("ONNX Runtime not found")

    print(f"\nStarting BLIP model export process...")
    print(f"Model ID: {model_id}")
    print(f"Output directory: {output_dir}")

    # Export model
    success, model_path = export_blip_to_onnx(model_id, output_dir)

    if success and model_path:
        # Need processor for verification
        try:
             processor_verify = BlipProcessor.from_pretrained(output_dir)
             # Verify the exported model
             verify_onnx_model(model_path, processor_verify)

             # List exported files
             print("\nExported files:")
             for file in os.listdir(output_dir):
                 print(f"- {file}")
        except Exception as e:
             print(f"Failed to load processor for verification or list files: {e}")
    else:
        print("Export process failed")

if __name__ == "__main__":
    main()