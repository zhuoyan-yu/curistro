import json

import httpx
from openai import OpenAI
import pytest

from curistro.models import Node
from curistro.providers import DemoProvider, LiveProvider, ProviderError

CONFIG = dict(CHAT_MODEL='o4-mini', QUIZ_MODEL='o4-mini', TTS_MODEL='tts-1', STT_MODEL='whisper-1',
              TTS_VOICE='alloy', OUTPUT_MODE='json_schema', TOKEN_PARAMETER='max_completion_tokens',
              MAX_OUTPUT_TOKENS=4096)


def node():
    return Node('owner', 'Topic', 'Demo', [], conversation=[{'role': 'user', 'message': 'A complete explanation.'}])


def adapter(handler, **overrides):
    client = OpenAI(api_key='test-only-key', base_url='https://provider.invalid/v1', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return LiveProvider({**CONFIG, **overrides}, client=client)


def completion(content, reason='stop'):
    return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'fixture',
        'choices': [{'index': 0, 'finish_reason': reason, 'message': {'role': 'assistant', 'content': content}}]})


def test_actual_sdk_request_uses_reasoning_compatible_parameters():
    def handler(request):
        payload = json.loads(request.content)
        assert payload['max_completion_tokens'] == 4096
        assert 'max_tokens' not in payload and 'temperature' not in payload
        assert 'A complete explanation.' in payload['messages'][1]['content']
        return completion('What is an example?')
    assert adapter(handler).question(node()) == 'What is an example?'


@pytest.mark.parametrize('output_mode', ['json_schema', 'json_object'])
def test_structured_quiz_validates_with_real_sdk(output_mode):
    def handler(request):
        payload = json.loads(request.content)
        assert payload['response_format']['type'] == output_mode
        return completion(DemoProvider().quiz(node()).model_dump_json())
    assert len(adapter(handler, OUTPUT_MODE=output_mode).quiz(node()).questions) == 4


def test_invalid_structured_response_is_an_error():
    provider = adapter(lambda request: completion('{"questions": []}'))
    with pytest.raises(ProviderError, match='invalid'):
        provider.quiz(node())


@pytest.mark.parametrize('content,reason', [('', 'stop'), ('partial', 'length'), ('blocked', 'content_filter'), ('partial', None)])
def test_empty_and_truncated_completions_are_errors(content, reason):
    with pytest.raises(ProviderError):
        adapter(lambda request: completion(content, reason)).question(node())


def test_provider_exception_does_not_disclose_raw_message():
    provider = adapter(lambda request: httpx.Response(401, json={'error': {'message': 'PRIVATE_CREDENTIAL_OR_PROMPT', 'type': 'invalid_request_error'}}))
    with pytest.raises(ProviderError) as exc:
        provider.question(node())
    assert 'PRIVATE_CREDENTIAL' not in str(exc.value)


def test_sdk_stream_handles_empty_choices_and_verbatim_unicode_chunks():
    events = [
        {'id': 'f', 'created': 0, 'model': 'f', 'object': 'chat.completion.chunk', 'choices': []},
        *[{'id': 'f', 'created': 0, 'model': 'f', 'object': 'chat.completion.chunk', 'choices': [
            {'index': 0, 'delta': {'content': text}, 'finish_reason': None}]} for text in ['Hel', 'lo\n', '你好']],
        {'id': 'f', 'created': 0, 'model': 'f', 'object': 'chat.completion.chunk', 'choices': [
            {'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
    ]
    def handler(request):
        assert json.loads(request.content)['stream'] is True
        body = ''.join('data: ' + json.dumps(item) + '\n\n' for item in events) + 'data: [DONE]\n\n'
        return httpx.Response(200, content=body.encode(), headers={'Content-Type': 'text/event-stream'})
    assert ''.join(adapter(handler).question_stream(node())) == 'Hello\n你好'


def test_answer_key_is_not_sent_to_student():
    question = DemoProvider().quiz(node()).questions[0]
    def handler(request):
        prompt = json.loads(request.content)['messages'][1]['content']
        question_json = prompt.split('Question:\n', 1)[1].split('\nReturn only JSON', 1)[0]
        assert 'answer' not in json.loads(question_json)
        return completion('{"choice": null, "rationale": "Insufficient information"}')
    assert adapter(handler).answer(node(), question).choice is None


def test_stream_without_completion_marker_is_rejected():
    chunk = {'id': 'f', 'created': 0, 'model': 'f', 'object': 'chat.completion.chunk', 'choices': [
        {'index': 0, 'delta': {'content': 'Partial answer'}, 'finish_reason': None}]}
    provider = adapter(lambda request: httpx.Response(200,
        content=('data: ' + json.dumps(chunk) + '\n\n').encode(),
        headers={'Content-Type': 'text/event-stream'}))
    with pytest.raises(ProviderError, match='before completion'):
        list(provider.question_stream(node()))


def test_audio_requests_use_original_recording_bytes():
    def handler(request):
        if request.url.path.endswith('/audio/speech'):
            body = json.loads(request.content)
            assert body['response_format'] == 'mp3' and body['voice'] == 'alloy'
            return httpx.Response(200, content=b'ID3-fixture', headers={'Content-Type': 'audio/mpeg'})
        assert request.url.path.endswith('/audio/transcriptions')
        assert b'filename="recording.webm"' in request.content and b'original-audio-bytes' in request.content
        return httpx.Response(200, json={'text': 'Recognized words'})
    provider = adapter(handler)
    assert provider.speak('Words') == b'ID3-fixture'
    assert provider.transcribe('recording.webm', b'original-audio-bytes', 'audio/webm') == 'Recognized words'
