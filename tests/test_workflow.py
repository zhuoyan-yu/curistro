import io
import json

import pytest
from pydantic import ValidationError

from curistro import create_app
from curistro.exports import ExportError, markdown_export, pdf_export
from curistro.models import Answer, Node, Quiz, Store, clarifier_prompt, evaluate
from curistro.providers import DemoProvider, ProviderError
from curistro.web import ROOT


@pytest.fixture
def app():
    return create_app({'TESTING': True, 'SECRET_KEY': 'local-test-key', 'MODE': 'demo'})


def token(client):
    client.get('/')
    with client.session_transaction() as session:
        return session['csrf']


def start(client, **values):
    data = dict(title='Photosynthesis', content='Plants use light to store energy in sugars.', category='Science', related='', csrf_token=token(client))
    data.update(values)
    response = client.post('/start_teach', data=data)
    assert response.status_code == 302
    return response.headers['Location'].rsplit('/', 1)[-1]


def post(client, url, **kwargs):
    return client.post(url, headers={'X-CSRF-Token': token(client), 'Accept': 'application/json'}, **kwargs)


def node_for(app, node_id):
    return app.extensions['curistro_store'].nodes[node_id]


def frames(response):
    return [(frame.splitlines()[0].removeprefix('event: '), json.loads(frame.splitlines()[1].removeprefix('data: ')))
            for frame in response.get_data(as_text=True).strip().split('\n\n')]


def test_complete_demo_workflow_and_exports(app):
    client = app.test_client()
    related = start(client, title='Energy')
    node_id = start(client, related='Energy')
    assert related != node_id
    assert client.get('/conversation/' + node_id).status_code == 200
    response = post(client, '/reply/' + node_id, json={'message': 'For example, a leaf in sunlight.'})
    assert frames(response)[-1][0] == 'done'
    assert post(client, '/explain/' + node_id, json={}).status_code == 200
    assert post(client, '/finish/' + node_id, json={}).status_code == 200
    node = node_for(app, node_id)
    assert node.report['correct'] == 3 and node.report['total'] == 4
    assert b'AI STUDENT CHECKS' in client.get('/result/' + node_id).data
    assert post(client, '/recap/' + node_id, json={}).json['recap']
    graph = client.get('/map_data').json
    assert graph['links'] == [{'source': node_id, 'target': related}]
    markdown = client.get('/summary/' + node_id + '?fmt=md')
    assert markdown.status_code == 200 and b'scripted sample' in markdown.data
    pdf = client.get('/summary/' + node_id + '?fmt=pdf')
    assert pdf.status_code == 200 and pdf.data.startswith(b'%PDF')


def test_missing_or_wrong_csrf_rejected(app):
    client = app.test_client()
    client.get('/')
    assert client.post('/start_teach', data={'title': 'bad'}).status_code == 403
    assert client.post('/start_teach', headers={'X-CSRF-Token': 'wrong'}).status_code == 403


def test_sessions_are_isolated_in_pages_map_and_mutations(app):
    a, b = app.test_client(), app.test_client()
    node_id = start(a, title='Private A topic')
    b.get('/')
    assert b.get('/map_data').json == {'links': [], 'nodes': []}
    for endpoint in ['conversation', 'result', 'summary']:
        assert b.get('/' + endpoint + '/' + node_id).status_code == 404
    for endpoint in ['reply', 'explain', 'finish', 'recap', 'speak', 'transcribe']:
        assert post(b, '/' + endpoint + '/' + node_id, json={}).status_code == 404


def test_same_title_nodes_are_not_overwritten(app):
    client = app.test_client()
    first, second = start(client), start(client)
    assert first != second and len(app.extensions['curistro_store'].nodes) == 2


def test_last_three_messages_are_preserved():
    node = Node('owner', 'Topic', 'Demo', [])
    node.conversation = [{'role': 'user', 'message': text} for text in ['OLD', 'First complete message', 'Second complete message', 'Third complete message']]
    prompt = clarifier_prompt(node)
    assert 'OLD' not in prompt
    for text in ['First complete message', 'Second complete message', 'Third complete message']:
        assert text in prompt


@pytest.mark.parametrize('mutation', ['empty', 'missing_answer', 'invalid_answer', 'missing_option', 'duplicate_dimension', 'blank_option'])
def test_invalid_quizzes_cannot_be_scored(mutation):
    node = Node('owner', 'Topic', 'Demo', [], conversation=[{'role': 'user', 'message': 'Text'}])
    provider = DemoProvider()
    data = provider.quiz(node).model_dump()
    if mutation == 'empty': data['questions'] = []
    elif mutation == 'missing_answer': del data['questions'][0]['answer']
    elif mutation == 'invalid_answer': data['questions'][0]['answer'] = ''
    elif mutation == 'missing_option': data['questions'][0]['options'].pop()
    elif mutation == 'duplicate_dimension': data['questions'][1]['dimension'] = data['questions'][0]['dimension']
    elif mutation == 'blank_option': data['questions'][0]['options'][0] = '  '
    provider.quiz = lambda node: data
    with pytest.raises(ValidationError): evaluate(node, provider)
    assert node.report is None and node.quiz is None


def test_brackets_inside_valid_json_question_survive():
    node = Node('owner', 'Lists', 'Demo', [], conversation=[{'role': 'user', 'message': 'Python lists'}])
    data = DemoProvider().quiz(node).model_dump()
    data['questions'][0]['options'][0] = '[1, 2]'
    quiz = Quiz.model_validate(data)
    assert len(quiz.questions) == 4 and quiz.questions[0].options[0] == '[1, 2]'


def test_provider_failure_never_creates_success_report(app, monkeypatch):
    client = app.test_client()
    node_id = start(client)
    def fail(*args): raise ProviderError('Simulated provider failure')
    monkeypatch.setattr(app.extensions['curistro_provider'], 'quiz', fail)
    response = post(client, '/finish/' + node_id, json={})
    assert response.status_code == 502
    assert node_for(app, node_id).report is None
    assert post(client, '/recap/' + node_id, json={}).status_code == 409


def test_invalid_or_failed_answer_rolls_back_whole_evaluation(app, monkeypatch):
    client = app.test_client()
    node_id = start(client)
    calls = []
    def answer(node, question):
        calls.append(question)
        return Answer(choice='A', rationale='Example') if len(calls) < 3 else {'choice': '', 'rationale': 'Invalid'}
    monkeypatch.setattr(app.extensions['curistro_provider'], 'answer', answer)
    assert post(client, '/finish/' + node_id, json={}).status_code == 502
    assert node_for(app, node_id).report is None and node_for(app, node_id).quiz is None


def test_abstention_is_not_a_correct_answer():
    node = Node('owner', 'Topic', 'Demo', [], conversation=[{'role': 'user', 'message': 'Text'}])
    provider = DemoProvider()
    provider.answer = lambda *args: Answer(choice=None, rationale='Insufficient explanation')
    assert evaluate(node, provider)['correct'] == 0


def test_stream_preserves_chunks_whitespace_and_unicode(app, monkeypatch):
    client = app.test_client()
    node_id = start(client)
    monkeypatch.setattr(app.extensions['curistro_provider'], 'question_stream', lambda node: iter(['Hel', 'lo', '\n你好', ' world']))
    response = post(client, '/reply/' + node_id, json={'message': 'More detail'})
    tokens = ''.join(data['text'] for event, data in frames(response) if event == 'token')
    assert tokens == 'Hello\n你好 world'
    assert node_for(app, node_id).conversation[-1]['message'] == tokens


def test_failed_stream_is_not_saved_and_can_retry(app, monkeypatch):
    client = app.test_client()
    node_id = start(client)
    provider = app.extensions['curistro_provider']
    original = provider.question_stream
    def fail(node):
        yield 'Partial'
        raise ProviderError('Stream failed')
    monkeypatch.setattr(provider, 'question_stream', fail)
    response = post(client, '/reply/' + node_id, json={'message': 'More detail'})
    assert frames(response)[-1][0] == 'error'
    node = node_for(app, node_id)
    assert node.conversation[-1] == {'role': 'user', 'message': 'More detail'}
    assert not node.lock.locked()
    monkeypatch.setattr(provider, 'question_stream', original)
    response = post(client, '/reply/' + node_id, json={'message': ''})
    assert frames(response)[-1][0] == 'done'
    assert sum(m['message'] == 'More detail' for m in node.conversation) == 1


def test_disconnected_stream_releases_lock(app, monkeypatch):
    client = app.test_client()
    node_id = start(client)
    response = post(client, '/reply/' + node_id, json={'message': 'New explanation'}, buffered=False)
    iterator = iter(response.response)
    next(iterator)
    response.close()
    assert not node_for(app, node_id).lock.locked()


def test_busy_node_rejects_concurrent_operations(app):
    client = app.test_client()
    node_id = start(client)
    node = node_for(app, node_id)
    with node.lock:
        assert post(client, '/finish/' + node_id, json={}).status_code == 409
        response = post(client, '/reply/' + node_id, json={'message': 'Must not append'})
        assert frames(response)[0][0] == 'error'
    assert all(m['message'] != 'Must not append' for m in node.conversation)


def test_state_guards_and_report_invalidation(app):
    client = app.test_client()
    node_id = start(client)
    assert client.get('/result/' + node_id).status_code == 409
    assert client.get('/summary/' + node_id).status_code == 409
    assert post(client, '/recap/' + node_id, json={}).status_code == 409
    post(client, '/finish/' + node_id, json={})
    post(client, '/reply/' + node_id, json={'message': 'An additional point'})
    assert node_for(app, node_id).report is None
    assert client.get('/summary/' + node_id).status_code == 409


def test_finish_is_cached_until_teaching_changes(app, monkeypatch):
    client = app.test_client()
    node_id = start(client)
    post(client, '/finish/' + node_id, json={})
    def unexpected(node): raise AssertionError('Repeated API charge')
    monkeypatch.setattr(app.extensions['curistro_provider'], 'quiz', unexpected)
    assert post(client, '/finish/' + node_id, json={}).status_code == 200


def test_untrusted_content_is_escaped_in_html_and_markdown(app):
    client = app.test_client()
    payload = '<img src=x onerror=alert(1)>'
    node_id = start(client, title=payload, content=payload)
    response = client.get('/conversation/' + node_id)
    assert payload.encode() not in response.data and b'&lt;img' in response.data
    post(client, '/finish/' + node_id, json={})
    response = client.get('/summary/' + node_id + '?fmt=md')
    assert payload.encode() not in response.data and b'&lt;img' in response.data
    assert "script-src 'self'" in response.headers['Content-Security-Policy']


def test_pdf_missing_glyph_is_explicit_not_silent(app):
    client = app.test_client()
    node_id = start(client, title='光合作用')
    post(client, '/finish/' + node_id, json={})
    response = client.get('/summary/' + node_id + '?fmt=pdf', headers={'Accept': 'application/json'})
    assert response.status_code == 422 and 'font' in response.json['error']
    assert client.get('/summary/' + node_id + '?fmt=md').status_code == 200


def test_initial_ai_failure_preserves_user_explanation(app, monkeypatch):
    client = app.test_client()
    def fail(node): raise ProviderError('Initial question unavailable')
    monkeypatch.setattr(app.extensions['curistro_provider'], 'question', fail)
    node_id = start(client)
    node = node_for(app, node_id)
    assert node.content and node.last_error and len(node.conversation) == 1


@pytest.mark.parametrize('field,value', [('title', ''), ('content', ' '), ('category', ''), ('title', 'x' * 121), ('content', 'x' * 8001)])
def test_invalid_input_rejected_without_creating_node(app, field, value):
    client = app.test_client()
    data = {'title': 'Topic', 'content': 'Explanation', 'category': 'Demo', 'csrf_token': token(client), field: value}
    assert client.post('/start_teach', data=data).status_code == 400
    assert not app.extensions['curistro_store'].nodes


def test_live_mode_requires_key_and_demo_never_needs_it(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        create_app({'MODE': 'live', 'API_KEY': ''})
    assert isinstance(create_app({'MODE': 'demo', 'API_KEY': ''}).extensions['curistro_provider'], DemoProvider)


def test_audio_passthrough_and_validation_without_ffmpeg():
    class AudioProvider(DemoProvider):
        def speak(self, text): return b'ID3-audio-fixture'
        def transcribe(self, filename, data, mime):
            assert filename == 'recording.webm' and data == b'fixture' and mime == 'audio/webm'
            return 'Transcribed explanation'
    app = create_app({'TESTING': True, 'SECRET_KEY': 'test', 'MODE': 'live'}, provider=AudioProvider())
    client = app.test_client()
    node_id = start(client)
    assert post(client, '/speak/' + node_id).data == b'ID3-audio-fixture'
    response = post(client, '/transcribe/' + node_id, data={'file': (io.BytesIO(b'fixture'), 'capture.webm')})
    assert response.json['text'] == 'Transcribed explanation'
    for filename, data in [('recording.exe', b'bytes'), ('empty.webm', b'')]:
        assert post(client, '/transcribe/' + node_id, data={'file': (io.BytesIO(data), filename)}).status_code == 400


def test_demo_audio_and_oversized_upload_rejected(app):
    client = app.test_client()
    node_id = start(client)
    assert post(client, '/speak/' + node_id).status_code == 409
    app.config['MAX_CONTENT_LENGTH'] = 1024
    assert post(client, '/start_teach', data={'content': 'x' * 2048}).status_code == 413


def test_no_model_calls_on_get_requests(app, monkeypatch):
    client = app.test_client()
    node_id = start(client)
    def fail(node): raise AssertionError('GET must not call the provider')
    monkeypatch.setattr(app.extensions['curistro_provider'], 'question', fail)
    for path in ['/', '/conversation/' + node_id, '/map_data']:
        assert client.get(path).status_code == 200
    assert client.get('/explain/' + node_id).status_code == 405
