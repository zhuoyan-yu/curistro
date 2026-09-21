"""Local Flask application with isolated browser sessions and explicit failures."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import secrets

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, session, stream_with_context, url_for
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from .exports import ExportError, markdown_export, pdf_export
from .models import Node, Store, evaluate
from .providers import DemoProvider, LiveProvider, ProviderError

ROOT = Path(__file__).resolve().parent.parent


def create_app(config=None, provider=None):
    load_dotenv(ROOT / '.env', override=False)
    app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=str(ROOT / 'static'))
    app.config.from_mapping(
        SECRET_KEY=os.getenv('CURISTRO_SECRET_KEY') or secrets.token_hex(32),
        MODE=os.getenv('CURISTRO_MODE', 'demo'), API_KEY=os.getenv('OPENAI_API_KEY', ''),
        BASE_URL=os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
        CHAT_MODEL=os.getenv('CURISTRO_CHAT_MODEL', 'o4-mini'),
        QUIZ_MODEL=os.getenv('CURISTRO_QUIZ_MODEL') or os.getenv('CURISTRO_CHAT_MODEL', 'o4-mini'),
        TTS_MODEL=os.getenv('CURISTRO_TTS_MODEL', 'tts-1'), STT_MODEL=os.getenv('CURISTRO_STT_MODEL', 'whisper-1'),
        TTS_VOICE=os.getenv('CURISTRO_TTS_VOICE', 'alloy'),
        OUTPUT_MODE=os.getenv('CURISTRO_OUTPUT_MODE', 'json_schema'),
        TOKEN_PARAMETER=os.getenv('CURISTRO_TOKEN_PARAMETER', 'max_completion_tokens'),
        MAX_OUTPUT_TOKENS=int(os.getenv('CURISTRO_MAX_OUTPUT_TOKENS', '4096')),
        PDF_FONT=os.getenv('CURISTRO_PDF_FONT', ''),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
        SESSION_COOKIE_NAME='curistro_session', TRUSTED_HOSTS=['localhost', '127.0.0.1', '[::1]'],
    )
    app.config.update(config or {})
    if app.config['MODE'] not in {'demo', 'live'}:
        raise ValueError('CURISTRO_MODE must be demo or live')
    if app.config['OUTPUT_MODE'] not in {'json_schema', 'json_object'}:
        raise ValueError('CURISTRO_OUTPUT_MODE must be json_schema or json_object')
    if app.config['TOKEN_PARAMETER'] not in {'max_completion_tokens', 'max_tokens'}:
        raise ValueError('Invalid CURISTRO_TOKEN_PARAMETER')
    if not 256 <= app.config['MAX_OUTPUT_TOKENS'] <= 32768:
        raise ValueError('CURISTRO_MAX_OUTPUT_TOKENS must be between 256 and 32768')
    if provider is None:
        if app.config['MODE'] == 'live' and not app.config['API_KEY']:
            raise ValueError('Live mode needs OPENAI_API_KEY in .env. Use CURISTRO_MODE=demo without a key.')
        provider = DemoProvider() if app.config['MODE'] == 'demo' else LiveProvider(app.config)
    store = Store()
    app.extensions.update(curistro_store=store, curistro_provider=provider)

    @app.before_request
    def session_and_csrf():
        if request.endpoint == 'static':
            return
        session.setdefault('owner', secrets.token_hex(24))
        session.setdefault('csrf', secrets.token_hex(24))
        if request.method == 'POST':
            supplied = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token', '')
            if not secrets.compare_digest(supplied, session['csrf']):
                abort(403, 'The page session expired. Refresh the page and try again.')

    @app.context_processor
    def page_context():
        return {'mode': app.config['MODE'], 'csrf_token': session.get('csrf', '')}

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.endpoint != 'static':
            response.headers['Cache-Control'] = 'no-store'
        return response

    def fail(message, status):
        if request.is_json or request.headers.get('Accept') == 'application/json':
            return jsonify(error=message), status
        return render_template('error.html', message=message), status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return fail(str(exc.description), exc.code)

    @app.errorhandler(ProviderError)
    def provider_error(exc):
        return fail(str(exc), 502)

    @app.errorhandler(ValidationError)
    def invalid_model_output(exc):
        return fail('The AI response was invalid. No score was saved. Please retry.', 502)

    @app.errorhandler(ExportError)
    def export_error(exc):
        return fail(str(exc), 422)

    def get_node(node_id):
        node = store.get(session['owner'], node_id)
        if not node:
            abort(404, 'This session was not found. It may belong to another browser or have ended when the app restarted.')
        return node

    @contextmanager
    def locked(node):
        if not node.lock.acquire(blocking=False):
            abort(409, 'This session is busy. Wait for the current request, then retry.')
        try:
            yield
        finally:
            node.lock.release()

    def text_field(data, key, limit, required=True):
        text = data.get(key, '')
        if not isinstance(text, str):
            abort(400, f'{key} must be text.')
        text = text.strip()
        if (required and not text) or len(text) > limit or any(ord(c) < 32 and c not in '\n\r\t' for c in text):
            abort(400, f'{key} must contain {"1" if required else "0"} to {limit} characters of text.')
        return text

    @app.get('/')
    def index():
        return render_template('index.html', nodes=list(reversed(store.for_owner(session['owner']))))

    @app.post('/start_teach')
    def start_teach():
        title = text_field(request.form, 'title', 120)
        content = text_field(request.form, 'content', 8000)
        category = text_field(request.form, 'category', 80)
        related = text_field(request.form, 'related', 500, required=False)
        node = Node(session['owner'], title, category, [r.strip().lstrip('#') for r in related.split(',') if r.strip()])
        node.conversation.append({'role': 'user', 'message': content})
        try:
            store.add(node)
        except ValueError as exc:
            abort(429, str(exc))
        try:
            node.conversation.append({'role': 'ai', 'message': provider.question(node)})
        except ProviderError as exc:
            node.last_error = str(exc)
        return redirect(url_for('conversation', node_id=node.id))

    @app.get('/conversation/<node_id>')
    def conversation(node_id):
        return render_template('conversation.html', node=get_node(node_id))

    @app.post('/reply/<node_id>')
    def reply(node_id):
        node = get_node(node_id)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, 'Expected a JSON message.')
        message = text_field(payload, 'message', 4000, required=False)

        def events():
            with locked(node):
                if len(node.conversation) >= 80 or len(node.content) + len(message) > 32000:
                    yield event('error', {'message': 'This conversation is full. Start another topic.'})
                    return
                if message:
                    node.conversation.append({'role': 'user', 'message': message})
                    node.invalidate_report()
                node.last_error = None
                yield event('accepted', {'message': message})
                parts, size = [], 0
                try:
                    for part in provider.question_stream(node):
                        parts.append(part)
                        size += len(part)
                        if size > 20000:
                            raise ProviderError('The reply exceeded the local limit. Please retry.')
                        yield event('token', {'text': part})
                    answer = ''.join(parts)
                    if not answer.strip():
                        raise ProviderError('The AI returned an empty reply. Please retry.')
                    node.conversation.append({'role': 'ai', 'message': answer})
                    yield event('done', {})
                except ProviderError as exc:
                    node.last_error = str(exc)
                    yield event('error', {'message': str(exc)})

        def guarded_events():
            try:
                yield from events()
            except HTTPException as exc:
                yield event('error', {'message': str(exc.description)})
        return Response(stream_with_context(guarded_events()), mimetype='text/event-stream',
                        headers={'X-Accel-Buffering': 'no'})

    @app.post('/explain/<node_id>')
    def explain(node_id):
        node = get_node(node_id)
        with locked(node):
            if len(node.conversation) >= 80:
                abort(409, 'This conversation is full. Start another topic.')
            text = provider.explain(node)
            node.conversation.append({'role': 'ai', 'message': text})
            return jsonify(text=text)

    @app.post('/finish/<node_id>')
    def finish(node_id):
        node = get_node(node_id)
        with locked(node):
            if node.report is None:
                evaluate(node, provider)
        return jsonify(url=url_for('result', node_id=node.id))

    @app.get('/result/<node_id>')
    def result(node_id):
        node = get_node(node_id)
        if node.report is None:
            abort(409, 'Finish the AI student checks before opening the report.')
        return render_template('result.html', node=node)

    @app.post('/recap/<node_id>')
    def recap(node_id):
        node = get_node(node_id)
        with locked(node):
            if node.report is None:
                abort(409, 'Finish the AI student checks before generating a recap.')
            if node.recap is None:
                node.recap = (provider.recap(node) if node.report['correct'] < node.report['total'] else
                              'The AI student answered all four checks correctly. Try a new example or a harder contrast next.')
            return jsonify(recap=node.recap)

    @app.get('/map_data')
    def map_data():
        nodes = store.for_owner(session['owner'])
        by_title = {n.title.casefold(): n.id for n in nodes}
        ids = {n.id for n in nodes}
        links = set()
        for node in nodes:
            for name in node.related:
                target = by_title.get(name.casefold()) or (name if name in ids else None)
                if target and target != node.id:
                    links.add((node.id, target))
        return jsonify(nodes=[{'id': n.id, 'label': n.title, 'category': n.category} for n in nodes],
                       links=[{'source': a, 'target': b} for a, b in sorted(links)])

    @app.get('/summary/<node_id>')
    def summary(node_id):
        node = get_node(node_id)
        fmt = request.args.get('fmt', 'md')
        if fmt not in {'md', 'pdf'}:
            abort(400, 'Choose Markdown or PDF.')
        with locked(node):
            if node.report is None:
                abort(409, 'Finish the AI student checks before exporting.')
            data = markdown_export(node, app.config['MODE']).encode('utf-8') if fmt == 'md' else pdf_export(node, app.config['MODE'], ROOT, app.config['PDF_FONT'])
        return send_file(io.BytesIO(data), mimetype='text/markdown; charset=utf-8' if fmt == 'md' else 'application/pdf',
                         as_attachment=True, download_name=f'curistro-{node.id}.{fmt}')

    @app.post('/speak/<node_id>')
    def speak(node_id):
        node = get_node(node_id)
        if app.config['MODE'] == 'demo':
            abort(409, 'Audio requires a configured live provider.')
        with locked(node):
            replies = [m['message'] for m in node.conversation if m['role'] == 'ai']
            if not replies:
                abort(409, 'There is no AI reply to read yet.')
            if len(replies[-1]) > 4096:
                abort(400, 'This reply is too long for speech. Use the text version.')
            return Response(provider.speak(replies[-1]), mimetype='audio/mpeg')

    @app.post('/transcribe/<node_id>')
    def transcribe(node_id):
        node = get_node(node_id)
        if app.config['MODE'] == 'demo':
            abort(409, 'Audio requires a configured live provider.')
        with locked(node):
            upload = request.files.get('file')
            if not upload or not upload.filename:
                abort(400, 'Select an audio recording.')
            extension = Path(upload.filename).suffix.lower()
            formats = {'.webm': 'audio/webm', '.mp4': 'audio/mp4', '.m4a': 'audio/mp4',
                       '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.mpeg': 'audio/mpeg', '.mpga': 'audio/mpeg'}
            if extension not in formats:
                abort(400, 'Use a WebM, MP4, M4A, WAV, or MP3 recording.')
            data = upload.read()
            if not data:
                abort(400, 'The recording is empty. Please try again.')
            text = provider.transcribe('recording' + extension, data, formats[extension])
            if len(text) > 4000:
                abort(422, 'The transcript is too long. Record a shorter explanation.')
            return jsonify(text=text)

    return app


def event(name, payload):
    return f'event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'
