import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Use your own API Keys or env file
_ = load_dotenv('./config/config.env')

# Load API Keys & Model
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
#

model = AzureChatOpenAI(
    temperature=0.3,
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    verbose=True
    )

embeddings = AzureOpenAIEmbeddings(
    chunk_size=1,
    openai_api_version=os.getenv("EMBEDDINGS_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
    openai_api_key=os.getenv("EMBEDDINGS_API_KEY"),
    model=os.getenv("EMBEDDINGS_MODEL"),
    deployment=os.getenv("EMBEDDINGS_DEPLOYMENT")
    )
