import torch
import torch.onnx
from PIL import Image
import os
import numpy as np
import traceback
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
import onnxruntime as ort
from typing import List, Tuple, Dict
from torch.nn import Module

# --- Configuration ---
MODEL_ID = "Salesforce/blip-image-captioning-base"
OUTPUT_DIR = Path("./blip_manual_onnx_export")
IMG_SIZE = 384  # Define image size based on model config
MAX_LENGTH = 32  # Example max sequence length for text
OPSET_VERSION = 14  # ONNX Opset version
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_past_key_values(past_key_values, prefix="past_"):
    """Flattens nested past_key_values tuple for ONNX input/output names."""
    flattened = {}
    for i, layer_past in enumerate(past_key_values):
        flattened[f"{prefix}key_{i}"] = layer_past[0]  # Key tensor
        flattened[f"{prefix}value_{i}"] = layer_past[1]  # Value tensor
    return flattened

def create_dummy_past_key_values(config, batch_size=1, seq_len=1, device='cpu'):
    """Creates dummy past_key_values tensors based on model config."""
    num_layers = config.text_config.num_hidden_layers
    num_heads = config.text_config.num_attention_heads
    embed_dim_per_head = config.text_config.hidden_size // num_heads
    past_shape = (batch_size, num_heads, seq_len, embed_dim_per_head)
    dummy_past = []
    for _ in range(num_layers):
        dummy_past.append(
            (torch.randn(past_shape, device=device),  # Dummy Key
             torch.randn(past_shape, device=device))  # Dummy Value
        )
    return tuple(dummy_past)

# --- Verification Functions ---
def verify_vision_encoder(model_path: Path, pixel_values: torch.Tensor) -> bool:
    """Verifies the vision encoder ONNX model."""
    try:
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        inputs = {'pixel_values': pixel_values.detach().cpu().numpy()}  # Detach tensor
        outputs = session.run(None, inputs)
        print(f"Vision encoder verification: Output shape {outputs[0].shape}")
        return True
    except Exception as e:
        print(f"Vision encoder verification failed: {e}")
        return False

def verify_decoder_init(model_path: Path, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor,
                        encoder_attention_mask: torch.Tensor) -> bool:
    """Verifies the decoder initial pass ONNX model."""
    try:
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        inputs = {
            'input_ids': input_ids.detach().cpu().numpy(),  # Detach tensor
            'encoder_hidden_states': encoder_hidden_states.detach().cpu().numpy(),  # Detach tensor
            'encoder_attention_mask': encoder_attention_mask.detach().cpu().numpy()  # Detach tensor
        }
        outputs = session.run(None, inputs)
        print(f"Decoder init verification: Logits shape {outputs[0].shape}, Past key-values {len(outputs[1:])} tensors")
        return True
    except Exception as e:
        print(f"Decoder init verification failed: {e}")
        return False

def verify_decoder_past(model_path: Path, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor,
                        encoder_attention_mask: torch.Tensor, past_key_values: Dict[str, torch.Tensor]) -> bool:
    """Verifies the decoder with past ONNX model."""
    try:
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        inputs = {
            'input_ids': input_ids.detach().cpu().numpy(),  # Detach tensor
            'encoder_hidden_states': encoder_hidden_states.detach().cpu().numpy(),  # Detach tensor
            'encoder_attention_mask': encoder_attention_mask.detach().cpu().numpy(),  # Detach tensor
            **{k: v.detach().cpu().numpy() for k, v in past_key_values.items()}  # Detach tensors
        }
        outputs = session.run(None, inputs)
        print(f"Decoder past verification: Logits shape {outputs[0].shape}, Past key-values {len(outputs[1:])} tensors")
        return True
    except Exception as e:
        print(f"Decoder past verification failed: {e}")
        return False

# --- Wrapper for Text Decoder ---
class TextDecoderWrapper(Module):
    """Wrapper to control text decoder forward pass for ONNX export."""
    def __init__(self, text_decoder):
        super().__init__()
        self.text_decoder = text_decoder

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask, past_key_values=None):
        outputs = self.text_decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=False  # Ensure tuple output for ONNX
        )
        logits = outputs[0]
        past_key_values = outputs[1] if len(outputs) > 1 else None
        if past_key_values:
            return logits, *past_key_values
        return logits

# --- Main Export Function ---
def export_blip_components(model_id: str, output_dir: Path) -> bool:
    print(f"Starting export for model: {model_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- 1. Load Model and Processor ---
        print("Loading PyTorch model and processor...")
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        processor = BlipProcessor.from_pretrained(model_id)
        model.eval()
        config = model.config
        vision_config = config.vision_config
        text_config = config.text_config
        model.to(DEVICE)
        print(f"Model and processor loaded. Using device: {DEVICE}")

        # --- 2. Export Vision Encoder ---
        print("\nExporting Vision Encoder...")
        vision_encoder = model.vision_model
        vision_output_path = output_dir / "vision_encoder.onnx"
        dummy_pixel_values = torch.randn(1, vision_config.num_channels, IMG_SIZE, IMG_SIZE, device=DEVICE)

        torch.onnx.export(
            vision_encoder,
            (dummy_pixel_values,),
            str(vision_output_path),
            input_names=['pixel_values'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'last_hidden_state': {0: 'batch_size'}
            },
            opset_version=OPSET_VERSION,
            export_params=True
        )
        print(f"Vision Encoder saved to {vision_output_path}")
        vision_outputs = vision_encoder(dummy_pixel_values)
        dummy_image_embeds = vision_outputs.last_hidden_state
        print(f"Vision Encoder output shape: {dummy_image_embeds.shape}")

        # Verify Vision Encoder
        print("Verifying Vision Encoder...")
        if not verify_vision_encoder(vision_output_path, dummy_pixel_values):
            print("Vision Encoder verification failed.")
            return False

        # --- 3. Export Text Decoder ---
        print("\nExporting Text Decoder...")
        text_decoder = TextDecoderWrapper(model.text_decoder)
        decoder_init_path = output_dir / "decoder_init.onnx"
        decoder_past_path = output_dir / "decoder_with_past.onnx"

        # --- 3a. Decoder Init Export ---
        print("Exporting Decoder (Initial Pass)...")
        dummy_decoder_input_ids = torch.tensor([[text_config.bos_token_id]], dtype=torch.long, device=DEVICE)
        dummy_encoder_hidden_states = dummy_image_embeds
        dummy_encoder_attention_mask = torch.ones(dummy_encoder_hidden_states.shape[:2], dtype=torch.long, device=DEVICE)

        outputs_init = text_decoder(
            input_ids=dummy_decoder_input_ids,
            encoder_hidden_states=dummy_encoder_hidden_states,
            encoder_attention_mask=dummy_encoder_attention_mask
        )
        dummy_past_key_values = outputs_init[1:]  # Skip logits
        flat_past_init = flatten_past_key_values(dummy_past_key_values, "past_")
        output_names_init = ["logits"] + list(flat_past_init.keys())

        torch.onnx.export(
            text_decoder,
            (dummy_decoder_input_ids, dummy_encoder_hidden_states, dummy_encoder_attention_mask),
            str(decoder_init_path),
            input_names=['input_ids', 'encoder_hidden_states', 'encoder_attention_mask'],
            output_names=output_names_init,
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'encoder_hidden_states': {0: 'batch_size', 1: 'vision_sequence_length'},
                'encoder_attention_mask': {0: 'batch_size', 1: 'vision_sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'},
                **{name: {0: 'batch_size', 2: 'past_sequence_length'} for name in flat_past_init.keys()}
            },
            opset_version=OPSET_VERSION,
            export_params=True
        )
        print(f"Decoder (Initial Pass) saved to {decoder_init_path}")

        # Verify Decoder Init
        print("Verifying Decoder (Initial Pass)...")
        if not verify_decoder_init(decoder_init_path, dummy_decoder_input_ids, dummy_encoder_hidden_states,
                                  dummy_encoder_attention_mask):
            print("Decoder Init verification failed.")
            return False

        # --- 3b. Decoder With Past Export ---
        print("Exporting Decoder (With Past)...")
        dummy_past_input_ids = torch.tensor([[7592]], dtype=torch.long, device=DEVICE)  # Dummy token
        input_names_past = ['input_ids', 'encoder_hidden_states', 'encoder_attention_mask'] + list(flat_past_init.keys())
        flat_present_past = flatten_past_key_values(dummy_past_key_values, "present_")
        output_names_past = ["logits"] + list(flat_present_past.keys())

        dynamic_axes_past = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'encoder_hidden_states': {0: 'batch_size', 1: 'vision_sequence_length'},
            'encoder_attention_mask': {0: 'batch_size', 1: 'vision_sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'},
            **{name: {0: 'batch_size', 2: 'past_sequence_length'} for name in flat_past_init.keys()},
            **{name: {0: 'batch_size', 2: 'past_sequence_length'} for name in flat_present_past.keys()}
        }

        torch.onnx.export(
            text_decoder,
            (dummy_past_input_ids, dummy_encoder_hidden_states, dummy_encoder_attention_mask, dummy_past_key_values),
            str(decoder_past_path),
            input_names=input_names_past,
            output_names=output_names_past,
            dynamic_axes=dynamic_axes_past,
            opset_version=OPSET_VERSION,
            export_params=True
        )
        print(f"Decoder (With Past) saved to {decoder_past_path}")

        # Verify Decoder Past
        print("Verifying Decoder (With Past)...")
        if not verify_decoder_past(decoder_past_path, dummy_past_input_ids, dummy_encoder_hidden_states,
                                  dummy_encoder_attention_mask, flat_past_init):
            print("Decoder Past verification failed.")
            return False

        # --- 4. Save Processor ---
        print("\nSaving processor...")
        processor.save_pretrained(output_dir)
        print(f"Processor saved to {output_dir}")

        print("\n--- Manual Export Process Completed ---")
        return True

    except Exception as e:
        print(f"\n--- ERROR during manual export ---")
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return False

# --- Generation Loop Example ---
def run_generation_loop(image_path: str, output_dir: Path, max_length: int = 32) -> str:
    """Runs a generation loop using ONNX models for image captioning."""
    # Load processor
    processor = BlipProcessor.from_pretrained(output_dir)
    
    # Load ONNX sessions
    vision_session = ort.InferenceSession(str(output_dir / "vision_encoder.onnx"), providers=['CPUExecutionProvider'])
    decoder_init_session = ort.InferenceSession(str(output_dir / "decoder_init.onnx"), providers=['CPUExecutionProvider'])
    decoder_past_session = ort.InferenceSession(str(output_dir / "decoder_with_past.onnx"), providers=['CPUExecutionProvider'])

    # Preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].numpy()

    # Run vision encoder
    image_embeds = vision_session.run(None, {'pixel_values': pixel_values})[0]
    encoder_attention_mask = np.ones(image_embeds.shape[:2], dtype=np.int64)

    # Initialize generation
    generated_ids = [processor.tokenizer.bos_token_id]
    past_key_values = None

    for _ in range(max_length):
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
            past_key_values = {f"past_key_{i//2}" if i % 2 == 0 else f"past_value_{i//2}": outputs[i+1] for i in range(len(outputs[1:]))}
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
            past_key_values = {f"present_key_{i//2}" if i % 2 == 0 else f"present_value_{i//2}": outputs[i+1] for i in range(len(outputs[1:]))}

        # Get next token
        next_token_id = np.argmax(logits[:, -1, :], axis=-1).item()
        generated_ids.append(next_token_id)

        # Stop at EOS
        if next_token_id == processor.tokenizer.eos_token_id:
            break

    # Decode caption
    caption = processor.decode(generated_ids, skip_special_tokens=True)
    return caption

# --- Main Execution ---
if __name__ == "__main__":
    success = export_blip_components(MODEL_ID, OUTPUT_DIR)
    if success:
        print("\nRunning example generation loop...")
        # Replace with a path to a test image
        test_image_path = r"C:\Users\voutl\OneDrive\Pictures\ai.png"
        if os.path.exists(test_image_path):
            caption = run_generation_loop(test_image_path, OUTPUT_DIR)
            print(f"Generated caption: {caption}")
        else:
            print("Please provide a valid test image path to run the generation loop.")