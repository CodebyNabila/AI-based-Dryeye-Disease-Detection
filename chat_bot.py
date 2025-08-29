import streamlit as st
import google.generativeai as genai

import time

# Configure API key (replace with your actual API key)
genai.configure(api_key="AIzaSyBVYhHjcPQuBZTpq2TrkbBI2HOZlrh3rno")

# Create model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def GenerateResponse(input_text):
    """Generates a response using the Gemini model based on input text."""
    try:
        response = model.generate_content([
            "input: who are you",
            "output: I am a Dry Eye chatbot 👁️",
            "input: what all can you do?",
            "output: I can help you with any Eye related help 👁️‍🗨️",
            f"input: {input_text}",
            "output: ",
        ])
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit App
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Dry Eye Chatbot",
        page_icon="👁️",
        layout="centered",  # Centered layout for compact view
        initial_sidebar_state="collapsed"  # Hide sidebar for compact view
    )

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stApp {
                background-color: #f0f2f6;
                font-family: 'Arial', sans-serif;
                max-width: 600px;  /* Limit width for compact view */
                margin: auto;  /* Center the chatbox */
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stChatMessage {
                border-radius: 15px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stChatMessage.user {
                background-color: #d1e7dd;
                color: #0f5132;
                margin-left: 20%;
                border-bottom-right-radius: 5px;
            }
            .stChatMessage.assistant {
                background-color: #fff3cd;
                color: #856404;
                margin-right: 20%;
                border-bottom-left-radius: 5px;
            }
            .stChatInput {
                background-color: #ffffff;
                border-radius: 15px;
                padding: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                width: 100%;  /* Full width input */
            }
            .loading {
                display: inline-block;
                width: 50px;
                height: 50px;
                border: 5px solid #f3f3f3;
                border-radius: 50%;
                border-top: 5px solid #3498db;
                border-right: 5px solid #e74c3c;
                border-bottom: 5px solid #2ecc71;
                border-left: 5px solid #f1c40f;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .stButton>button {
                background-color: #3498db;
                color: white;
                border-radius: 15px;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stButton>button:hover {
                background-color: #2980b9;
            }
            .stTitle {
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="stTitle">👁️ Dry Eye Chatbot</div>', unsafe_allow_html=True)

    # Chatbot UI
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you with your eye concerns today? 👋"}]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask me anything about dry eyes..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Display loading animation
        with st.spinner(""):
            loading_placeholder = st.empty()
            loading_placeholder.markdown('<div class="loading"></div>', unsafe_allow_html=True)
            time.sleep(1)  # Simulate loading time

        # Get response from the model
        response = GenerateResponse(prompt)

        # Remove loading animation and display response
        loading_placeholder.empty()
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()