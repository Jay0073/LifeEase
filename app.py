import streamlit as st
from llama_cpp import Llama
import os
import time

def initialize_model():
    """Initialize the Gemma model with optimized settings"""
    os.environ['LLAMA_METAL_NDISABLE'] = '1'
    os.environ['LLAMA_NUMA'] = '1'
    os.environ['LLAMA_MMX_NTHREADS'] = '8'

    return Llama(
        model_path=r"C:\Users\voutl\OneDrive\Documents\LifeEase\gemma-2-2b-it-Q5_K_M.gguf",
        n_batch=512,
        n_threads=8,
        n_gpu_layers=-1,
        seed=42,
    )

def get_streaming_response(model, user_input, temperature, top_p, top_k, assistant_index, message_placeholder):
    """Generate streaming response from the model"""
    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    full_response = ""

    for chunk in model(
        prompt,
        max_tokens=150,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stream=True
    ):
        chunk_text = chunk["choices"][0]["text"]
        full_response += chunk_text

        # Update session state
        st.session_state.messages[assistant_index]["content"] = full_response

        # Update entire HTML inside the placeholder
        message_placeholder.markdown(f"""
            <div class="message-row">
                <div class="assistant-message">{full_response}▌</div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.01)

    # Final render without the cursor
    message_placeholder.markdown(f"""
        <div class="message-row">
            <div class="assistant-message">{full_response}</div>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Gemma Chatbot", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .css-1d391kg {
            background-color: #2d2d2d;
        }
        .stMarkdown h1 {
            color: #ffffff;
        }
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
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stTextInput input {
            background-color: #333333;
            color: white;
            border: 1px solid #555555;
            border-radius: 10px;
        }
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
        .stSlider {
            color: #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("💬 Gemma Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = initialize_model()

    st.sidebar.header("Model Parameters")
    temperature = st.sidebar.slider("Temperature (0.2 - 0.4 ideal)", 0.0, 1.0, 0.7)
    top_p = st.sidebar.slider("Top P (0.8 - 0.9 ideal)", 0.0, 1.0, 0.9)
    top_k = st.sidebar.slider("Top K (20 - 40 ideal)", 0.0, 1.0, 0.3)

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display existing chat history
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

    # Input from user
    user_input = st.chat_input("What would you like to ask?")
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Add empty assistant message to be filled during streaming
        st.session_state.messages.append({"role": "assistant", "content": ""})
        assistant_index = len(st.session_state.messages) - 1

        # Render user message immediately
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="message-row">
                <div class="user-message">{user_input}</div>
            </div>
        """, unsafe_allow_html=True)

        # Render empty assistant message with placeholder
        st.markdown(f"""
            <div class="message-row">
                <div class="assistant-message">
        """, unsafe_allow_html=True)
        message_placeholder = st.empty()
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Get and stream response
        get_streaming_response(
            model=st.session_state.model,
            user_input=user_input,
            temperature=temperature,
            top_p=int(top_p),
            top_k=int(top_k),
            assistant_index=assistant_index,
            message_placeholder=message_placeholder
        )

if __name__ == "__main__":
    main()
