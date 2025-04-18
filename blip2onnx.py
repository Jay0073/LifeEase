import torch
import torch.onnx
import numpy as np
import traceback
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
import onnxruntime as ort
from typing import Dict
from torch.nn import Module

# --- Configuration ---
MODEL_ID = "Salesforce/blip-image-captioning-large"
OUTPUT_DIR = Path("./blip_large_onnx")
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
        inputs = {'pixel_values': pixel_values.detach().cpu().numpy()}
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
            'input_ids': input_ids.detach().cpu().numpy(),
            'encoder_hidden_states': encoder_hidden_states.detach().cpu().numpy(),
            'encoder_attention_mask': encoder_attention_mask.detach().cpu().numpy()
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
            'input_ids': input_ids.detach().cpu().numpy(),
            'encoder_hidden_states': encoder_hidden_states.detach().cpu().numpy(),
            'encoder_attention_mask': encoder_attention_mask.detach().cpu().numpy(),
            **{k: v.detach().cpu().numpy() for k, v in past_key_values.items()}
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
            return_dict=False
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
        dummy_decoder_input_ids = torch.tensor([[text_config.bos_token_id or 0]], dtype=torch.long, device=DEVICE)
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
        dummy_past_input_ids = torch.tensor([[7592]], dtype=torch.long, device=DEVICE)
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

# --- Main Execution ---
if __name__ == "__main__":
    export_blip_components(MODEL_ID, OUTPUT_DIR)