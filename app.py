from imports import *
import streamlit as st
import requests
from PIL import Image
import logging 
import base64
from io import BytesIO

# Set up the Streamlit app configuration
st.set_page_config(page_title="Chatbot Application", layout="centered")

# Custom CSS to style the background and text elements
st.markdown(
    """
    <style>
    div {
        background-color: lightblue;
    }
    
    h1 {
        color: darkred; /* Change title color */
        text-align: center; /* Center the title */
    }
    .stTextInput div[role="textbox"] > div::after {
        background: none !important;
    """,
    unsafe_allow_html=True
)
# border:1px solid black;
logo = Image.open("usd-logo.png")  

# Convert image to base64 for embedding
buffered = BytesIO()
logo.save(buffered, format="PNG")
logo_base64 = base64.b64encode(buffered.getvalue()).decode()

# Use custom HTML for side-by-side alignment with base64 image
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 75px; margin-right: 10px;">
        <h1 style="margin: 0;">SQUAD Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)



# Display the title of the chatbot app with custom color
st.write("Ask me anything, and I'll try my best to answer!")

# get chat response
def get_ollama_response(input_text):
    logging.info(f"response: {input_text}")
    response = requests.post(
        "http://localhost:8001/",
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

 # Set up basic configuration for logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
   
embeddings = HuggingFaceEmbeddings()
db = Chroma(persist_directory="./", embedding_function=embeddings)

logging.info("initialize retriever")
retriever = db.as_retriever()


# Define the LLM and prompt template
logging.info("initialize model and prompt.")

llm = Ollama(model="llama2") 

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
<context>
{context} 
</context>
Question: {input}""")

logging.info(prompt)

# Create retrieval chains
logging.info("creating retrieval chains")
document_chain=create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

logging.info("initialize FAST api.")

# Define the FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Chatbot API Server"
)

@app.get("/")
async def read_root():
    return {"message": f"Welcome to SQUAD chatbot"}


@app.post("/")
async def get_response(input: dict):
    try:
        logging.info(f'got a request: {input}')

        query = (input.get('input')).get('input')
        logging.info(f"Received query: {query}")
        logging.info(retrieval_chain)
        response = retrieval_chain.invoke({"input": query})
        logging.info(response['answer'])
        if response:
            return {"answer": response['answer']}
        else:
            return {"answer": "No answer generated. Please try with a different query."}
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logging.info("Starting FastAPI server at http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8001)









