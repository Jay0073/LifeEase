import numpy as np
from PIL import Image
from pathlib import Path
import onnxruntime as ort
from transformers import BlipProcessor
import traceback

# --- Configuration ---
MODEL_DIR = Path("./blip_large_onnx")
VISION_MODEL_PATH = MODEL_DIR / "vision_encoder.onnx"
DECODER_INIT_PATH = MODEL_DIR / "decoder_init.onnx"
DECODER_PAST_PATH = MODEL_DIR / "decoder_with_past.onnx"
MAX_LENGTH = 64

def run_generation_loop(image_path: str, model_dir: Path, max_length: int = MAX_LENGTH) -> str:
    """Runs a generation loop using ONNX models for image captioning."""
    try:
        # Load processor
        print("Loading processor...")
        processor = BlipProcessor.from_pretrained(model_dir)
        print("Processor loaded.")

        # Load ONNX sessions
        print("Loading ONNX models...")
        vision_session = ort.InferenceSession(str(VISION_MODEL_PATH), providers=['CPUExecutionProvider'])
        decoder_init_session = ort.InferenceSession(str(DECODER_INIT_PATH), providers=['CPUExecutionProvider'])
        decoder_past_session = ort.InferenceSession(str(DECODER_PAST_PATH), providers=['CPUExecutionProvider'])
        print("ONNX models loaded.")

        # Preprocess image
        print("Preprocessing image...")
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].numpy()
        print("Image preprocessed.")

        # Run vision encoder
        print("Running vision encoder...")
        image_embeds = vision_session.run(None, {'pixel_values': pixel_values})[0]
        encoder_attention_mask = np.ones(image_embeds.shape[:2], dtype=np.int64)
        print("Vision encoder output shape:", image_embeds.shape)

        # Initialize generation
        # Use pad_token_id or 0 if bos_token_id is None
        start_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
        generated_ids = [start_token_id]
        past_key_values = None

        print("Starting generation loop...")
        for i in range(max_length):
            input_ids = np.array([generated_ids[-1]], dtype=np.int64).reshape(1, 1)

            if past_key_values is None:
                # Initial decoder pass
                inputs = {
                    'input_ids': input_ids,
                    'encoder_hidden_states': image_embeds,
                    'encoder_attention_mask': encoder_attention_mask
                }
                outputs = decoder_init_session.run(None, inputs)
                logits = outputs[0]
                past_key_values = {
                    f"past_key_{i//2}" if i % 2 == 0 else f"past_value_{i//2}": outputs[i+1]
                    for i in range(len(outputs[1:]))
                }
            else:
                # Subsequent decoder passes
                inputs = {
                    'input_ids': input_ids,
                    'encoder_hidden_states': image_embeds,
                    'encoder_attention_mask': encoder_attention_mask,
                    **past_key_values
                }
                outputs = decoder_past_session.run(None, inputs)
                logits = outputs[0]
                # Keep key names as 'past_*' to match model input expectations
                past_key_values = {
                    f"past_key_{i//2}" if i % 2 == 0 else f"past_value_{i//2}": outputs[i+1]
                    for i in range(len(outputs[1:]))
                }

            # Get next token
            next_token_id = np.argmax(logits[:, -1, :], axis=-1).item()
            generated_ids.append(next_token_id)

            # Stop at EOS or max length
            if processor.tokenizer.eos_token_id is not None and next_token_id == processor.tokenizer.eos_token_id:
                print(f"EOS token reached at step {i+1}")
                break
            if i == max_length - 1:
                print("Max length reached")
                break

        # Decode caption
        print("Decoding caption...")
        caption = processor.decode(generated_ids, skip_special_tokens=True)
        print("Caption generated successfully.")
        return caption

    except Exception as e:
        print(f"Error during generation: {e}")
        traceback.print_exc()
        return ""

# --- Main Execution ---
if __name__ == "__main__":
    # Update with your test image path
    test_image_path = r"C:\Users\voutl\OneDrive\Desktop\download (1).jpg"  # From your output
    if not Path(test_image_path).exists():
        print(f"Image path {test_image_path} does not exist. Please provide a valid image path.")
    else:
        print(f"Generating caption for image: {test_image_path}")
        caption = run_generation_loop(test_image_path, MODEL_DIR)
        if caption:
            print(f"Generated caption: {caption}")
        else:
            print("Failed to generate caption.")