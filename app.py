import os
import time
import json
import requests
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PLATFORM_URL = os.environ.get("PLATFORM_URL", "https://platform.skapa.design")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deepseek-r1")
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() in ("true", "1", "yes")

# Cache des clés API validées (dict Python avec TTL)
_key_cache = {}
KEY_CACHE_TTL = 300  # 5 minutes

http_session = requests.Session()

# ---------------------------------------------------------------------------
# Validation des clés API
# ---------------------------------------------------------------------------

def _validate_single_key(api_key: str) -> bool:
    """Valide une clé API auprès de la plateforme Skapa."""
    try:
        resp = http_session.post(
            f"{PLATFORM_URL}/api/v1/validate-single-key",
            json={"api_key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("valid", False)
        return False
    except Exception as e:
        print(f"Erreur validation clé : {e}")
        return False


def is_key_valid(api_key: str) -> bool:
    """Vérifie la validité d'une clé avec cache en mémoire (TTL 5 min)."""
    now = time.time()
    cached = _key_cache.get(api_key)
    if cached and (now - cached["ts"]) < KEY_CACHE_TTL:
        return cached["valid"]

    valid = _validate_single_key(api_key)
    _key_cache[api_key] = {"valid": valid, "ts": now}
    return valid


def get_api_key():
    """Extrait et valide la clé API du header Authorization."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None, (jsonify({"error": "Clé API manquante ou invalide"}), 401)
    key = auth[7:]
    if DEV_MODE:
        return key, None
    if not is_key_valid(key):
        return None, (jsonify({"error": "Clé API non autorisée"}), 403)
    return key, None


# ---------------------------------------------------------------------------
# Helpers Ollama
# ---------------------------------------------------------------------------

def _ollama_chat(payload: dict) -> requests.Response:
    """Envoie une requête chat à Ollama Cloud."""
    payload.setdefault("model", DEFAULT_MODEL)
    payload["stream"] = False  # synchrone pour cPanel/Passenger

    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    return http_session.post(
        f"{OLLAMA_BASE_URL}/chat",
        json=payload,
        headers=headers,
        timeout=120,
    )


def _ollama_generate(payload: dict) -> requests.Response:
    """Envoie une requête generate à Ollama Cloud."""
    payload.setdefault("model", DEFAULT_MODEL)
    payload["stream"] = False

    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    return http_session.post(
        f"{OLLAMA_BASE_URL}/generate",
        json=payload,
        headers=headers,
        timeout=120,
    )


def _openai_to_ollama_chat(openai_payload: dict) -> dict:
    """Convertit un payload OpenAI chat/completions vers le format Ollama."""
    messages = openai_payload.get("messages", [])
    return {
        "model": openai_payload.get("model", DEFAULT_MODEL),
        "messages": messages,
        "options": {
            "temperature": openai_payload.get("temperature", 0.7),
            "top_p": openai_payload.get("top_p", 1.0),
        },
    }


def _ollama_to_openai_chat(ollama_resp: dict, model: str) -> dict:
    """Convertit une réponse Ollama en format OpenAI."""
    message = ollama_resp.get("message", {})
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
            "completion_tokens": ollama_resp.get("eval_count", 0),
            "total_tokens": (
                ollama_resp.get("prompt_eval_count", 0)
                + ollama_resp.get("eval_count", 0)
            ),
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    key, err = get_api_key()
    if err:
        return err

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON body"}), 400

    model = payload.get("model", DEFAULT_MODEL)
    ollama_payload = _openai_to_ollama_chat(payload)

    try:
        resp = _ollama_chat(ollama_payload)
        resp.raise_for_status()
        result = _ollama_to_openai_chat(resp.json(), model)
        return jsonify(result)
    except requests.exceptions.Timeout:
        return jsonify({"error": "LLM timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Service LLM indisponible"}), 503
    except Exception as e:
        return jsonify({"error": f"Erreur interne du proxy : {e}"}), 500


@app.route("/v1/completions", methods=["POST"])
def completions():
    key, err = get_api_key()
    if err:
        return err

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON body"}), 400

    prompt = payload.get("prompt", "")
    model = payload.get("model", DEFAULT_MODEL)

    ollama_payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": payload.get("temperature", 0.7),
        },
    }

    try:
        resp = _ollama_generate(ollama_payload)
        resp.raise_for_status()
        data = resp.json()

        result = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": data.get("response", ""),
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            },
        }
        return jsonify(result)
    except requests.exceptions.Timeout:
        return jsonify({"error": "LLM timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Service LLM indisponible"}), 503
    except Exception as e:
        return jsonify({"error": f"Erreur interne du proxy : {e}"}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    key, err = get_api_key()
    if err:
        return err

    try:
        headers = {}
        if OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

        resp = http_session.get(
            f"{OLLAMA_BASE_URL}/tags",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        ollama_models = resp.json().get("models", [])

        openai_models = []
        for m in ollama_models:
            openai_models.append(
                {
                    "id": m.get("name", "unknown"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "skapa",
                }
            )

        return jsonify({"object": "list", "data": openai_models})
    except Exception as e:
        return jsonify({"error": f"Impossible de lister les modèles : {e}"}), 503


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": int(time.time())})


@app.route("/docs", methods=["GET"])
def docs():
    return """<!DOCTYPE html>
<html><head><title>Skapa API Docs</title>
<style>
body{font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;color:#333}
h1{color:#2563eb}h2{color:#1e40af;border-bottom:2px solid #e5e7eb;padding-bottom:8px}
code{background:#f3f4f6;padding:2px 6px;border-radius:4px;font-size:.9em}
pre{background:#1e293b;color:#e2e8f0;padding:16px;border-radius:8px;overflow-x:auto}
.method{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:700;font-size:.85em;margin-right:8px}
.post{background:#22c55e;color:#fff}.get{background:#3b82f6;color:#fff}
</style></head><body>
<h1>Skapa API Gateway</h1>
<p>Gateway compatible OpenAI. Authentification par clé API (Bearer token).</p>

<h2>Endpoints</h2>

<h3><span class="method post">POST</span><code>/v1/chat/completions</code></h3>
<p>Chat completion compatible OpenAI.</p>
<pre>curl -X POST https://api.skapa.design/v1/chat/completions \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"deepseek-r1","messages":[{"role":"user","content":"Bonjour"}]}'</pre>

<h3><span class="method post">POST</span><code>/v1/completions</code></h3>
<p>Text completion compatible OpenAI.</p>

<h3><span class="method get">GET</span><code>/v1/models</code></h3>
<p>Liste les modèles disponibles.</p>

<h3><span class="method get">GET</span><code>/health</code></h3>
<p>Healthcheck (pas d'auth requise).</p>

<h2>Authentification</h2>
<p>Ajoutez un header <code>Authorization: Bearer VOTRE_CLE_API</code> à chaque requête.</p>
<p>Obtenez votre clé sur <a href="https://platform.skapa.design">platform.skapa.design</a>.</p>
</body></html>"""


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
