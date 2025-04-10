from llama_cpp import Llama
import os
import gc  # Add this import for garbage collection

# Add these environment variables before model loading
os.environ['LLAMA_METAL_NDISABLE'] = '1'      # Disable metal acceleration if not needed
os.environ['LLAMA_NUMA'] = '1'                # Enable NUMA optimization
os.environ['LLAMA_MMX_NTHREADS'] = '8'        # Set number of MMX threads

try:
    # Model configuration for better performance
    model = Llama(
        model_path="gemma-2-2b-it-Q5_K_M.gguf",
        n_ctx=512,          # Reduce context window (default is 2048)
        n_batch=512,        # Increase batch size for better throughput
        n_threads=8,        # Adjust based on your CPU cores
        n_gpu_layers=-1,    # Keep GPU acceleration
        seed=42,            # Set seed for reproducibility
    )

    # Warm up the model with a dummy run
    _ = model("test", max_tokens=1)  # This will be slow but subsequent runs will be faster

    # Format prompt according to Gemma's template
    prompt = "<start_of_turn>user\nbut in some model cards, there are context lengths mentioned<end_of_turn>\n<start_of_turn>model\n"

    # With streaming
    model_answer = ""
    for chunk in model(prompt, 
                      max_tokens=500,
                      temperature=0.7,
                      top_p=0.95,
                      top_k=40,
                      stream=True):
        chunk_text = chunk["choices"][0]["text"]
        model_answer += chunk_text
        print(chunk_text, end="", flush=True)

finally:
    # Memory management
    if 'model' in locals():
        model.reset()                    # Reset model state
        del model                        # Delete model instance
    gc.collect()                         # Force garbage collection 