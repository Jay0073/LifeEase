from llama_cpp import Llama
import os
import gc
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# --- Setup Gemma with llama_cpp ---
def setup_gemma():
    os.environ['LLAMA_NUMA'] = '1'
    os.environ['LLAMA_MMX_NTHREADS'] = '8'

    model = Llama(
        model_path="gemma-2-2b-it-Q5_K_M.gguf",  # Ensure this path is correct
        n_ctx=512,
        n_threads=8,
        n_gpu_layers=-1 
    )
    return model

# --- Setup BLIP for Image Captioning ---
def setup_blip():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # Move to GPU if available, else CPU (mobile may need CPU-only)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        print(f"Error setting up BLIP: {e}")
        return None, None, None

# --- Generate Response (Text or Image-based) ---
def generate_response(gemma_model, blip_processor=None, blip_model=None, device=None, user_input=None, image_path=None):
    if image_path and blip_processor and blip_model:  # Image mode
        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = blip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                caption_ids = blip_model.generate(**inputs, max_length=50)
            caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
            print(f"BLIP Caption: {caption}")
        except Exception as e:
            print(f"Error processing image: {e}")
            caption = "Image processing failed."

        # Format prompt with caption for Gemma
        prompt = f"<start_of_turn>user\nProvide a detailed description of this scene: {caption}<end_of_turn>\n<start_of_turn>model\n"
    elif user_input:  # Text mode
        prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    else:
        return "Please provide text or an image."

    # Generate streaming response with Gemma
    response = ""
    try:
        for chunk in gemma_model(
            prompt,
            max_tokens=500,
            temperature=0.7,
            stream=True
        ):
            text_chunk = chunk["choices"][0]["text"]
            response += text_chunk
            print(text_chunk, end="", flush=True)
    except Exception as e:
        print(f"Error during generation: {e}")
    
    return response

# --- Cleanup Resources ---
def cleanup(gemma_model, blip_model=None):
    # Cleanup Gemma
    gemma_model.reset()
    del gemma_model
    
    # Cleanup BLIP if loaded
    if blip_model:
        del blip_model
        torch.cuda.empty_cache()  # Clear GPU memory if used
    
    gc.collect()

# --- Main Function ---
def main():
    try:
        # Initialize models
        gemma_model = setup_gemma()
        blip_processor, blip_model, device = setup_blip()

        # Example: Image input
        print("\n--- Image Example ---")
        image_path = r"C:\Users\voutl\OneDrive\Pictures\Screenshots\Screenshot (2).png"  # Replace with your image path
        response = generate_response(
            gemma_model,
            blip_processor,
            blip_model,
            device,
            image_path=image_path
        )

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Cleanup
        cleanup(gemma_model, blip_model)

if __name__ == "__main__":
    main()