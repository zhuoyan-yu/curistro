# Curistro — Learning by Teaching

Curistro lets you learn by teaching an AI student. Explain a concept, answer its clarifying questions, and see how well it can apply what you taught.

Built by **Zhuoyan Yu** and **Xiongfeng Li** as team **AI Pathfinders** for the **NYU Shanghai Digital Innovation Challenge** (December 2024–April 2025). **Best Innovation Award.**

## How it works

1. **Teach a concept.** Start with an explanation, then develop it through the AI student's questions.
2. **Hear it back.** Ask the student to explain the idea in its own words. Speech input and text-to-speech are available with a configured AI provider.
3. **Review its understanding.** The AI answers questions across four dimensions: Term Explanation, Component Analysis, Application, and Concept Contrast. A follow-up recap suggests what to explain next.
4. **Connect and keep your work.** Link related topics on a knowledge map and export the session as Markdown or PDF.

Quiz scores describe the **AI student's answers**. The knowledge map connects the related topics you enter.

## Versions

- **[dic-2025](https://github.com/zhuoyan-yu/curistro/tree/dic-2025)** preserves the competition demo's application code and functionality. Credentials and temporary files are excluded.
- **main** contains the current local version, with updated setup, session handling, and provider configuration. Subsequent maintenance is by Zhuoyan Yu.

## Try it locally

Requires **Python 3.12**. From the project directory:

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

**macOS / Linux**

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open **http://127.0.0.1:5058/** and select **Use an example** to walk through a session.

The default **demo mode uses scripted responses and sample scores**. It works without an API key; speech features require live mode.

## Connect an AI provider

Copy `.env.example` to `.env` and set:

```dotenv
CURISTRO_MODE=live
OPENAI_API_KEY=your-own-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

Choose models available from your provider using `CURISTRO_CHAT_MODEL`, `CURISTRO_QUIZ_MODEL`, `CURISTRO_TTS_MODEL`, and `CURISTRO_STT_MODEL`. Restart the app after changing configuration. `.env` is ignored by Git.

The default configuration uses strict JSON output and `max_completion_tokens`. Compatible providers can use `CURISTRO_OUTPUT_MODE=json_object` or `CURISTRO_TOKEN_PARAMETER=max_tokens` when required. If an answer exceeds the output budget, adjust `CURISTRO_MAX_OUTPUT_TOKENS`.

Speech input transcribes a completed recording of up to 60 seconds into editable text. Speech output uses an AI-generated voice. Live mode sends text and requested recordings to your configured provider; API charges may apply.

## Session storage and exports

The app runs locally and keeps sessions in memory. Tabs in the same browser share a session; separate browser profiles have separate sessions. Export anything you want to keep before restarting. There are no accounts or persistent storage.

Markdown supports Unicode text. PDF uses DejaVu Sans; for Chinese or other unsupported characters, set `CURISTRO_PDF_FONT` to a suitable local font or choose Markdown.

## Development

Install `requirements-dev.txt` in the virtual environment, then run `python -m pytest`. Tests cover the teaching workflow, session isolation, provider responses, streaming, and exports. Provider requests are mocked; live model and audio behavior depend on your configuration.

The backend is in `curistro/`, the interface in `templates/` and `static/`. GitHub Actions runs the tests on Windows and Linux.

See [third-party resources](docs/THIRD_PARTY.md) for D3 and DejaVu licenses.
