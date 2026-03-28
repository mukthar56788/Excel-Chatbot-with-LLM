from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mixtral-8x7b-instruct-v0.1",
    huggingfacehub_api_token=os.getenv(""),
    temperature=0.7
)
print(llm.invoke("Hello, test!"))
