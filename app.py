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
    
    # Custom CSS for dark theme and enhanced chat styling
    st.markdown(
        """
        <style>
        /* Dark theme for the entire app */
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2d2d2d;
        }
        
        /* Header styling */
        .stMarkdown h1 {
            color: #ffffff;
        }
        
        /* Chat container styling */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }
        
        .message-row {
            width: 100%;
            display: flex;
            margin-bottom: 10px;
        }
        
        /* User message styling */
        .user-message {
            background-color: #4CAF50;
            color: white;
            border-radius: 15px 15px 0 15px;
            padding: 12px 18px;
            margin-left: auto;
            max-width: 70%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            animation: fadeIn 0.3s ease-in;
        }
        
        /* Assistant message styling */
        .assistant-message {
            background-color: #424242;
            color: #ffffff;
            border-radius: 15px 15px 15px 0;
            padding: 12px 18px;
            margin-right: auto;
            max-width: 70%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            animation: fadeIn 0.3s ease-in;
        }
        
        /* Animation for messages */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Styling for input box */
        .stTextInput input {
            background-color: #333333;
            color: white;
            border: 1px solid #555555;
            border-radius: 10px;
        }
        
        /* Styling for buttons */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            transition: background-color 0.3s;
        }
        
        .stButton button:hover {
            background-color: #45a049;
        }
        
        /* Slider styling */
        .stSlider {
            color: #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("💬 Gemma Chatbot")

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

    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="message-row">
                    <div class="user-message">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="message-row">
                    <div class="assistant-message">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Handle user input
    user_input = st.chat_input("What would you like to ask?")
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Create a new container for the assistant's response
        response_container = st.container()
        
        with response_container:
            message_placeholder = st.empty()
            
            # Get and stream response
            response = get_streaming_response(
                model=st.session_state.model,
                user_input=user_input,
                temperature=temperature,
                max_tokens=max_tokens,
                message_placeholder=message_placeholder
            )
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the chat history properly
            st.rerun()

if __name__ == "__main__":
    main()