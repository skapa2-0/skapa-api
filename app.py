import os
import time
import hashlib
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'skapa-api-secret')

PLATFORM_URL = os.environ.get('PLATFORM_URL', 'https://platform.skapa.design')
OLLAMA_API_URL = os.environ.get('OLLAMA_API_URL', 'https://ollama.com/api')
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY', '')
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'deepseek-v3.2')

VALIDATED_KEYS = {}
KEY_CACHE_TTL = 300


def validate_api_key(auth_header):
    if not auth_header or not auth_header.startswith('Bearer '):
        return None, 'Clé API manquante ou format invalide'

    key = auth_header[7:]
    cache_key = hashlib.sha256(key.encode()).hexdigest()
    now = time.time()

    if cache_key in VALIDATED_KEYS:
        entry = VALIDATED_KEYS[cache_key]
        if now - entry['ts'] < KEY_CACHE_TTL:
            return entry['user'], None

    try:
        resp = requests.post(
            f'{PLATFORM_URL}/api/v1/validate-single-key',
            json={'key_to_validate': key},
            timeout=10,
            verify=False
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('valid'):
                user_info = data.get('user', {})
                VALIDATED_KEYS[cache_key] = {'user': user_info, 'ts': now}
                return user_info, None
            return None, data.get('error', 'Clé invalide')
        return None, f'Validation échouée (HTTP {resp.status_code})'
    except requests.RequestException as e:
        if cache_key in VALIDATED_KEYS:
            return VALIDATED_KEYS[cache_key]['user'], None
        return None, f'Service de validation indisponible: {e}'


def report_usage_to_platform(api_key, model, prompt_tokens, completion_tokens, response_time_ms=None, client_ip=None):
    """Fire-and-forget usage reporting to the platform"""
    try:
        requests.post(
            f'{PLATFORM_URL}/api/v1/report-usage',
            json={
                'api_key': api_key,
                'model': model,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'status_code': 200,
                'response_time_ms': response_time_ms,
                'client_ip': client_ip
            },
            timeout=5,
            verify=False
        )
    except Exception:
        pass


def call_ollama(model, messages, temperature=0.7, max_tokens=2048):
    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens
        }
    }
    headers = {'Content-Type': 'application/json'}
    if OLLAMA_API_KEY:
        headers['Authorization'] = f'Bearer {OLLAMA_API_KEY}'

    resp = requests.post(
        f'{OLLAMA_API_URL}/chat',
        json=payload,
        headers=headers,
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    auth_header = request.headers.get('Authorization')
    user, error = validate_api_key(auth_header)
    if error:
        return jsonify({'error': {'message': error, 'type': 'authentication_error'}}), 401

    data = request.get_json()
    if not data or 'messages' not in data:
        return jsonify({'error': {'message': 'messages field required', 'type': 'invalid_request'}}), 400

    model = data.get('model', DEFAULT_MODEL)
    messages = data['messages']
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 2048)

    start_time = time.time()
    try:
        result = call_ollama(model, messages, temperature, max_tokens)
    except requests.HTTPError as e:
        return jsonify({'error': {'message': f'LLM backend error: {e}', 'type': 'server_error'}}), 502
    except requests.RequestException as e:
        return jsonify({'error': {'message': f'LLM backend unavailable: {e}', 'type': 'server_error'}}), 503
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    response_message = result.get('message', {})
    content = response_message.get('content', '')
    thinking = response_message.get('thinking', '')
    if not content and thinking:
        content = thinking

    prompt_tokens = result.get('prompt_eval_count', 0)
    completion_tokens = result.get('eval_count', 0)

    openai_response = {
        'id': f'chatcmpl-{int(time.time())}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'index': 0,
            'message': {
                'role': response_message.get('role', 'assistant'),
                'content': content
            },
            'finish_reason': result.get('done_reason', 'stop')
        }],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }
    }

    api_key = auth_header[7:] if auth_header else ''
    report_usage_to_platform(
        api_key=api_key,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_time_ms=elapsed_ms,
        client_ip=request.remote_addr
    )

    return jsonify(openai_response)


@app.route('/v1/completions', methods=['POST'])
def completions():
    auth_header = request.headers.get('Authorization')
    user, error = validate_api_key(auth_header)
    if error:
        return jsonify({'error': {'message': error, 'type': 'authentication_error'}}), 401

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': {'message': 'prompt field required', 'type': 'invalid_request'}}), 400

    model = data.get('model', DEFAULT_MODEL)
    prompt = data['prompt']
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 2048)

    messages = [{'role': 'user', 'content': prompt}]

    start_time = time.time()
    try:
        result = call_ollama(model, messages, temperature, max_tokens)
    except requests.HTTPError as e:
        return jsonify({'error': {'message': f'LLM backend error: {e}', 'type': 'server_error'}}), 502
    except requests.RequestException as e:
        return jsonify({'error': {'message': f'LLM backend unavailable: {e}', 'type': 'server_error'}}), 503
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    response_message = result.get('message', {})
    prompt_tokens = result.get('prompt_eval_count', 0)
    completion_tokens = result.get('eval_count', 0)

    openai_response = {
        'id': f'cmpl-{int(time.time())}',
        'object': 'text_completion',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'text': response_message.get('content', ''),
            'index': 0,
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }
    }

    api_key = auth_header[7:] if auth_header else ''
    report_usage_to_platform(
        api_key=api_key,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_time_ms=elapsed_ms,
        client_ip=request.remote_addr
    )

    return jsonify(openai_response)


AVAILABLE_MODELS = [
    {'id': 'deepseek-v3.2', 'description': 'Raisonnement avancé'},
    {'id': 'qwen3-coder-next', 'description': 'Coding agent'},
    {'id': 'devstral-2:123b', 'description': 'Coding 123B'},
    {'id': 'devstral-small-2:24b', 'description': 'Coding 24B'},
    {'id': 'gemma3:27b', 'description': 'Google Gemma 3'},
    {'id': 'glm-5', 'description': 'Raisonnement 744B'},
    {'id': 'minimax-m2.5', 'description': 'Coding/productivité'},
]


@app.route('/v1/models', methods=['GET'])
def list_models():
    user, error = validate_api_key(request.headers.get('Authorization'))
    if error:
        return jsonify({'error': {'message': error, 'type': 'authentication_error'}}), 401

    models_list = [{
        'id': m['id'],
        'object': 'model',
        'created': 1700000000,
        'owned_by': 'skapa',
        'description': m['description']
    } for m in AVAILABLE_MODELS]

    return jsonify({'object': 'list', 'data': models_list})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': int(time.time()), 'version': '1.0.0'})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Skapa API',
        'version': '1.0.0',
        'description': 'Gateway API compatible OpenAI — Skapa',
        'endpoints': {
            '/v1/chat/completions': 'POST — Chat completions (OpenAI-compatible)',
            '/v1/completions': 'POST — Text completions',
            '/v1/models': 'GET — List available models',
            '/health': 'GET — Health check',
        },
        'authentication': 'Bearer token (get your API key at platform.skapa.design)',
        'documentation': 'https://platform.skapa.design/api-docs/'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
