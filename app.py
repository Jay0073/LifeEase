import streamlit as st
from llama_cpp import Llama
import os
import time

def initialize_model():
    """Initialize the Gemma model with optimized settings"""
    # Set environment variables
    os.environ['LLAMA_METAL_NDISABLE'] = '1'
    os.environ['LLAMA_NUMA'] = '1'
    os.environ['LLAMA_MMX_NTHREADS'] = '8'

    # Initialize and return the model
    return Llama(
        model_path=r"C:\Users\voutl\OneDrive\Documents\LifeEase\gemma-2-2b-it-Q5_K_M.gguf",
        n_batch=512,
        n_threads=8,
        n_gpu_layers=-1,
        seed=42,
    )

def get_streaming_response(model, user_input, temperature, max_tokens, message_placeholder):
    """Generate streaming response from the model"""
    # Format prompt according to Gemma's template
    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"

    full_response = ""

    # Stream the response
    for chunk in model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        stream=True
    ):
        chunk_text = chunk["choices"][0]["text"]
        full_response += chunk_text
        # Show streaming text with cursor
        message_placeholder.markdown(full_response + "▌")
        time.sleep(0.01)

    # Show final response without cursor
    message_placeholder.markdown(full_response)
    return full_response

def main():
    # Page configuration
    st.set_page_config(page_title="Gemma Chatbot", layout="wide")
    st.title("💬 Gemma Chatbot")

    # Custom CSS for WhatsApp-like chat
    st.markdown(
        """
        <style>
        .stChatMessage.user {
            background-color: #dcf8c6; /* Light green for user messages */
            border-radius: 10px;
            padding: 8px 15px;
            margin-bottom: 5px;
            float: right;
            clear: both;
            text-align: right;
        }
        .stChatMessage.assistant {
            background-color: #f0f0f0; /* Light grey for assistant messages */
            border-radius: 10px;
            padding: 8px 15px;
            margin-bottom: 5px;
            float: left;
            clear: both;
            text-align: left;
        }
        .stChatMessage div {
            white-space: pre-wrap; /* Preserve line breaks */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = initialize_model()

    # Sidebar parameters
    st.sidebar.header("Model Parameters")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 256)

    # Add clear chat button to sidebar
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Display chat history in correct order (newest at bottom)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    user_input = st.chat_input("What would you like to ask?")
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get model response
        with st.chat_message("assistant"):
            # Create placeholder for this specific response
            message_placeholder = st.empty()

            # Show thinking indicator in the placeholder
            message_placeholder.markdown("🤔 Thinking...")

            # Get and stream response using the same placeholder
            response = get_streaming_response(
                model=st.session_state.model,
                user_input=user_input,
                temperature=temperature,
                max_tokens=max_tokens,
                message_placeholder=message_placeholder
            )

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Rerun to update the chat history cleanly
        st.rerun()

if __name__ == "__main__":
    main()