from imports import *

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






