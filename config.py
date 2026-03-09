import os
from dotenv import load_dotenv

load_dotenv()

PLATFORM_URL = os.environ.get("PLATFORM_URL", "https://platform.skapa.design")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deepseek-r1")
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() in ("true", "1", "yes")
