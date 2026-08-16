from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm_cerebras = ChatCerebras(
    model_name="gpt-oss-120b"
    )

llm_groq = ChatGroq(
    model_name="llama-3.1-8b-instant"
)