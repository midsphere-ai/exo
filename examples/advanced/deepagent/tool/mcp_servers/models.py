from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv(verbose=True)
import os
import httpx

class ModelManager():
    def __init__(self):
        self.registered_models = {}

    def init_models(self):
        self._register_gemini()
        self._register_openai()

    def _register_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY")
        os.environ['GEMINI_API_KEY'] = api_key
        model_id = 'gemini-2.5-flash'
        model = ChatGoogleGenerativeAI(
            model = model_id,
            google_api_key = api_key, 
            temperature=0.2,
            max_output_tokens = 1024,
            timeout=60,
            http_client = httpx.Client(verify=False),
            http_async_client = httpx.AsyncClient(verify=False)
        )
        self.registered_models[model_id] = model
    
    def _register_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        os.environ['OPENAI_API_KEY'] = api_key
        model_id = 'gpt-5'
        base_url = os.getenv('OPENAI_BASE_URL')
        model = ChatOpenAI(
            model = model_id, 
            api_key = api_key,
            base_url = base_url,
            timeout = 30,
            max_retries=2,
            http_client = httpx.Client(verify=False),
            http_async_client = httpx.AsyncClient(verify=False)
        )
        self.registered_models[model_id] = model