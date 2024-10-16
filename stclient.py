import streamlit as st
import requests
from PIL import Image
import logging  # Import the logging module

# Set up the Streamlit app configuration
st.set_page_config(page_title="Chatbot Application", page_icon="🤖", layout="centered")

# Custom CSS to style the background and text elements
st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    .main {
        background-color: lightblue;
    }
    .stTextInput, .stTextArea {
        background-color: white;
    }
    .stButton button {
        background-color: darkblue;
        color: white;
    }
    .stButton button:hover {
        background-color: navy;
        color: white;
    }
    .title {
        color: darkred; /* Change title color */
        text-align: center; /* Center the title */
    }
    .center-logo {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Center the logo by using the custom CSS class
logo = Image.open("usd-logo.png")  # Replace with your logo path
st.markdown('<div class="center-logo">', unsafe_allow_html=True)
st.image(logo, width=200)
st.markdown('</div>', unsafe_allow_html=True)

# Display the title of the chatbot app with custom color
st.markdown('<h1 class="title">Chatbot Application</h1>', unsafe_allow_html=True)
st.write("Ask me anything, and I'll try my best to answer!")


# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def get_ollama_response(input_text):
    logging.info(f"response: {input_text}")
    response = requests.post(
        "http://localhost:8000/",
        json={'input': {'input': input_text}}
    )
    logging.info("I am here")
    if response.status_code != 200:
            logging.error(f"Request failed with status: {response.status_code}")
            return f"Error: {response.status_code}. {response.json().get('detail', 'No additional information')}"
    response_json = response.json().get('answer')
    return response_json

# Get user inputs
input_text = st.text_input("Ask a question")

# Show response for input_text
if input_text:
    with st.spinner("Generating response..."):
        st.write(get_ollama_response(input_text))
                