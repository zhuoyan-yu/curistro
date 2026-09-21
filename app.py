import eventlet
eventlet.monkey_patch() 
import os, time, hashlib, re, json, numpy
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, Response, stream_with_context
from openai import OpenAI  
from flask import jsonify, send_file, make_response
import markdown, io, pdfkit, shlex
try:
    import weasyprint                 #  for PDF export (pip install weasyprint)
except:
    weasyprint = None                 #  PDF will be disabled gracefully
import tempfile
from fpdf import FPDF
import openai
import io, tempfile, os, logging
from pydub import AudioSegment  
# point pydub at your ffmpeg binary:
AudioSegment.converter = r"c:\ffmpeg\ffmpeg\bin\ffmpeg.exe"

# --------------------------
# Load environment variables
notebook_dir = Path(os.getcwd())
env_path = notebook_dir / "api.env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ API key not found. Check your api.env file.")
print("API key loaded:", api_key[:6] + "****")

# Initialize OpenAI client with openai endpoint
client = OpenAI(api_key=api_key, base_url="https://api.gptsapi.net/v1")

# --------------------------
# Define data structures and AI classes

class KnowledgeNode:
    def __init__(self, title, content, category, related=None):
        self.id = self.generate_id(title)
        self.title = title
        self.content = content
        self.category = category
        self.related = related or []
        self.weight = 1.0
        self.last_review = time.time()
        self.error_count = 0
        self.generated_quiz = None
        self.quiz_report = None
        self.conversation = []  # Store conversation messages as a list of dicts

    def generate_id(self, title):
        timestamp = int(time.time())
        title_hash = hashlib.md5(title.encode()).hexdigest()[:6]
        return f"{timestamp}{title_hash}"  # No '#' for URL safety

class StudentAI:
    # ⇨ OPENAI  – model picks in one place
    CHAT_MODEL      = "o4-mini"     # good quality, cheaper than GPT-4-Turbo
    QUIZ_MODEL      = "o4-mini"     # same; change to gpt-4o for max quality
    TEMPERATURE     = 1.0
    def __init__(self):
        self.knowledge_base = {}
        self.title_index = {}  

    def teach(self, title, content, category, related):
        node = KnowledgeNode(title, content, category, related)
        self.knowledge_base[node.id] = node
        slug = title.lower().strip()
        self.title_index[slug] = node.id
        return node

    def get_node(self, node_id):
        return self.knowledge_base.get(node_id)

    def ask_openai(self, prompt, model=CHAT_MODEL):
        try:
            print("[Debug] Sending prompt to API:", prompt)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.TEMPERATURE,
                max_tokens=1024
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return "⚠️ API response format error: no choices found."
        except Exception as e:
            return f"Error calling API: {str(e)}"

    def _clarifier_prompt(self, explanation, history, title):
        recent = "\n".join(f"- {m['message']}" for m in history if m['role']=='user')[-3:]
        return (
            f"You are **Curistro-Student**, a curious beginner learning **{title}**.\n\n"
            "CONTEXT (latest teacher turns):\n"
            f"{recent}\n\n"
            "TASK: Ask *exactly one* clarifying question that would help you understand the explanation better.\n"
            "CONSTRAINTS:\n"
            "• Do **NOT** answer your own question.\n"
            "• Keep it ≤ 20 words.\n"
            "• Don’t quote the text verbatim – paraphrase the bit you’re unclear about.\n\n"
            f"EXPLANATION:\n{explanation}"
        )
    
    def ask_openai_stream(self, prompt, model=CHAT_MODEL):
        rsp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=self.TEMPERATURE,
            stream=True
        )
        for chunk in rsp:
            text = chunk.choices[0].delta.content
            if text:                         #  ➜ drop empty pieces
                yield text

    def generate_ai_student_response(self, node):
        """
        Generate a clarifying question from the AI student,
        based on the last teacher message.
        """
        if not node:
            return "Please start your teaching session."
        return self.ask_openai(
            self._clarifier_prompt(node.conversation[-1]['message'],
                                node.conversation,
                                node.title), model=self.CHAT_MODEL
        )

    def ai_question_stream(self, node):
        prompt = self._clarifier_prompt(node.conversation[-1]['message'],
                                        node.conversation,
                                        node.title)
        return self.ask_openai_stream(prompt, model=self.CHAT_MODEL)
    
# ⇨ MOD START — new helpers inside StudentAI
    # ----------  EXPLAIN-BACK  ----------
    def explain_back(self, conversation):
        """
        AI student restates the concept in its own words and flags any
        remaining confusion.
        """
        whole = "\n".join(msg['message'] for msg in conversation if msg['role'] == 'user')
        prompt = (
          "You are Curistro-Student. Restate the topic below in your own words "
          "so the teacher can judge your understanding. "
          "After the paraphrase, list any points that still feel unclear.\n\n"
          f"EXPLANATION TO PARAPHRASE:\n{whole}"
        )
        return self.ask_openai(prompt, model=self.CHAT_MODEL)

    def generate_recap(self, node, weak_dims):
        """
        Build a micro-lesson that focuses ONLY on the dimensions the learner
        missed in the most recent quiz.
        """
        if not weak_dims:
            return None                                    # 100 % score → no recap

        dims_md = ", ".join(weak_dims)                    # e.g. "Component Analysis, Concept Contrast"
        bullet_dims = "\n".join(f"- {d}" for d in weak_dims)

        prompt = f"""
You are **Curistro-Coach**.

Learner’s topic: **{node.title}**
Quiz dimensions that were answered incorrectly:
{bullet_dims}

Write a SHORT micro-lesson (≤ 200 words) that:

1. Starts with the bold heading “Micro-Lesson: {dims_md}”.
2. Contains **one subsection per weak dimension** (use bold labels).
3. Gives an *analogy or example that references the topic* “{node.title}”.
4. Ends with a single practice task asking the learner to apply BOTH / ALL of the listed dimensions.

No extra commentary.
"""
        return self.ask_openai(prompt, model=self.QUIZ_MODEL)

    def generate_quiz(self, node):
        """
        Generate a quiz based strictly on the knowledge stored in the node.
        Uses aggregated teacher messages from the conversation.
        """
        aggregated_text = node.content  # node.content is updated to include all teacher messages.
        prompt = f"""
You are **Curistro-Judge**. Write **exactly four** multiple-choice questions (options A–D) designed to diagnose how well a learner understands the topic below.

TOPIC TITLE: {node.title}
EXPLANATION GIVEN BY LEARNER:
\"\"\"{aggregated_text}\"\"\"

Cover each of these diagnostic dimensions once **and in this order**:
1. Term Explanation
2. Component Analysis
3. Application
4. Concept Contrast

OUTPUT FORMAT (strict, no extra text):
[Dimension 1] Term Explanation
Question text…
A) Option one
B) Option two
C) Option three
D) Option four
Correct Answer: B

<blank line>

[…repeat for Dimensions 2-4…]
"""
        response = self.ask_openai(prompt, model=self.QUIZ_MODEL)
        return self._parse_quiz_response(response)

    def _parse_quiz_response(self, text):
        """
        Parse the API response into a structured quiz format.
        This parser now maps dimension labels to the desired ones.
        """
        dimensions_mapping = {
            "Dimension 1": "Term Explanation",
            "Dimension 2": "Component Analysis",
            "Dimension 3": "Application",
            "Dimension 4": "Concept Contrast",
            "Term Explanation": "Term Explanation",
            "Component Analysis": "Component Analysis",
            "Application": "Application",
            "Concept Contrast": "Concept Contrast"
        }
        quiz_data = {'questions': [], 'framework': ""}
        current_question = None
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            # Handle lines that start with "Dimension" (e.g., "Dimension 1: Question: ...")
            if stripped.startswith("Dimension"):
                if current_question:
                    quiz_data['questions'].append(current_question)
                parts = stripped.split("Question:", 1)
                dim_label = parts[0].replace(":", "").strip()  # e.g., "Dimension 1"
                question_text = parts[1].strip() if len(parts) > 1 else ""
                mapped_dimension = dimensions_mapping.get(dim_label, dim_label)
                current_question = {
                    'en_dim': dim_label,
                    'zh_dim': mapped_dimension,  # Using the updated desired label
                    'question': question_text,
                    'options': [],
                    'answer': ''
                }
            # Also support a bracketed format if provided
            elif ('[' in stripped and ']' in stripped) or ('【' in stripped and '】' in stripped):
                if current_question:
                    quiz_data['questions'].append(current_question)
                dim_match = re.search(r'\[(.*?)\]', stripped)
                if dim_match:
                    en_dim = dim_match.group(1).strip()
                else:
                    continue
                mapped_dimension = dimensions_mapping.get(en_dim, en_dim)
                current_question = {
                    'en_dim': en_dim,
                    'zh_dim': mapped_dimension,
                    'question': "",
                    'options': [],
                    'answer': ''
                }
            elif re.match(r'^[A-D][\).]', stripped):
                option_text = stripped[2:].strip()  # Remove letter and punctuation
                if current_question is not None:
                    current_question['options'].append(option_text)
            elif stripped.startswith("Correct Answer"):
                parts = stripped.split(":", 1)
                if len(parts) > 1 and current_question is not None:
                    # Clean up the answer by removing non-alphanumeric characters and forcing uppercase.
                    current_question['answer'] = re.sub(r'\W', '', parts[1].strip().upper())
            else:
                if current_question:
                    if current_question['question']:
                        current_question['question'] += " " + stripped
                    else:
                        current_question['question'] = stripped
        if current_question:
            quiz_data['questions'].append(current_question)
        return quiz_data


    def auto_evaluate(self, node):
        """
        Evaluate the AI student's performance by asking quiz questions
        and comparing the expected answers.
        """
        if not node.generated_quiz:
            return None
        report = {
            "total": len(node.generated_quiz['questions']),
            "correct": 0,
            "details": []
        }
        for q in node.generated_quiz['questions']:
            llm_answer = self._get_llm_answer(q['question'], q['options'])
            # Clean up both expected and obtained answers (should be single letters)
            expected = re.sub(r'\W', '', q['answer'].upper())
            obtained = re.sub(r'\W', '', (llm_answer['choice'] or "").upper())
            print(f"[Debug] Expected: {expected}, Obtained: {obtained}")  # Debug info for each question
            is_correct = (obtained == expected)
            report['correct'] += int(is_correct)
            detail = {**q, **llm_answer, "correct": is_correct}
            report['details'].append(detail)
        return report

    def _get_llm_answer(self, question, options):
        formatted_options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        analysis = self.answer_question(question, formatted_options)
        print("[Debug] AI analysis output:", analysis)
        # Remove Markdown bold formatting and extraneous asterisks.
        analysis_cleaned = re.sub(r'\*\*', '', analysis)
        # Use a regex that allows for spaces and newline characters between the indicator and the letter.
        answer_match = re.search(r"(?:Suggested Answer:|Recommended Answer:)[\s\r\n]*([A-D])", analysis_cleaned, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer_choice = answer_match.group(1).upper()
        else:
            print("[Debug] No recommended answer letter was found.")
            answer_choice = None
        print(f"[Debug] Extracted answer: {answer_choice}")
        return {"analysis": analysis, "confidence": {}, "choice": answer_choice}



    def answer_question(self, question, options):
        context = self.generate_context(question)
        knowledge_keywords = re.findall(r"#\w+", context)
        if not knowledge_keywords:
            return "This question is beyond my knowledge base."
        prompt = f"""You are an AI that must answer questions strictly based on the following knowledge base:

{context}

Rules:
1. If the question involves unknown content, answer: "I don't know [specific knowledge point]".
2. Analyze the relevance of each option to the knowledge base.
3. Finally, provide a recommended answer.

Options: {options}
"""
        return self.ask_openai(prompt, model=self.QUIZ_MODEL)

    def generate_context(self, question):
        context = ["Current Knowledge Base:"]
        # Include each node with a '#' prefix for proper matching.
        for node in self.knowledge_base.values():
            context.append(f"#{node.id} {node.title} ({node.weight:.1f}): {node.content}")
        context.append(f"\nQuestion: {question}")
        return "\n".join(context)

# --------------------------
# Initialize Flask application

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key

# Create a global StudentAI instance
student_ai = StudentAI()

# --------------------------
# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_teach', methods=['POST'])
def start_teach():
    title = request.form['title']
    content = request.form['content']
    category = request.form['category']
    related = request.form.get('related', '').split(',')
    related = [r.strip() for r in related if r.strip()]
    node = student_ai.teach(title, content, category, related)
    # Add the teacher's initial explanation to the conversation history.
    node.conversation.append({"role": "user", "message": content})
    return redirect(url_for('conversation', node_id=node.id))

@app.route('/conversation/<node_id>', methods=['GET', 'POST'])
def conversation(node_id):
    node = student_ai.get_node(node_id)
    if not node:
        return "Knowledge node not found", 404
    if request.method == 'POST':
        user_message = request.form['message']
        node.conversation.append({"role": "user", "message": user_message})
        ai_response = student_ai.generate_ai_student_response(node)
        node.conversation.append({"role": "ai", "message": ai_response})
        return redirect(url_for('conversation', node_id=node.id))
    # On GET: If only the initial teacher message exists, auto-generate the first AI clarifying question.
    teacher_messages = [msg for msg in node.conversation if msg['role'] == 'user']
    ai_messages = [msg for msg in node.conversation if msg['role'] == 'ai']
    if len(teacher_messages) == 1 and not ai_messages:
        ai_response = student_ai.generate_ai_student_response(node)
        node.conversation.append({"role": "ai", "message": ai_response})
        return redirect(url_for('conversation', node_id=node.id))
    return render_template('conversation.html', node=node)

from flask import Response, stream_with_context

# ---------- 1. record the teacher message ----------
@app.route('/send/<node_id>', methods=['POST'])
def send(node_id):
    node = student_ai.get_node(node_id)
    if not node:
        return "Node not found", 404

    user_msg = request.form['message'].strip()
    node.conversation.append({"role": "user", "message": user_msg})

    # launch the stream in a background green-thread so it is ready
    node._stream_iter = student_ai.ai_question_stream(node)
    return '', 204                                   # no content, return fast


# ---------- 2. stream the AI reply ----------
@app.route('/stream/<node_id>')
def stream(node_id):
    node = student_ai.get_node(node_id)
    if not node or not hasattr(node, '_stream_iter'):
        return "Node not ready", 404

    def event_stream():
        buf, prev_char = [], ''
        for raw_tok in node._stream_iter:
            # decide if we need to prefix a space
            need_space = (
                prev_char and not prev_char.isspace() and raw_tok
                and raw_tok[0] not in ".,;:!?)]}’”'\""
            )
            tok = (' ' if need_space else '') + raw_tok
            buf.append(tok)
            prev_char = tok[-1]
            yield f"data:{tok}\n\n"
        full = ''.join(buf)
        node.conversation.append({"role": "ai", "message": full})
        yield "event:done\ndata:1\n\n"

    return Response(stream_with_context(event_stream()),
                    mimetype='text/event-stream')

# ⇨ MOD START — new flask route, place near other routes
@app.route('/explain/<node_id>')
def explain(node_id):
    node = student_ai.get_node(node_id)
    if not node:
        return "Node not found", 404
    text = student_ai.explain_back(node.conversation)
    # log it in history
    node.conversation.append({"role": "ai", "message": text})
    return jsonify(text=text)
# ⇨ MOD END

# ---------- Text-to-Speech ----------
@app.route('/speak', methods=['POST'])
def speak():
    text = request.json.get('text','')[:4096]
    rsp  = client.audio.speech.create(
        model="tts-1",
        input=text,
        voice="alloy"
    )
    # return raw bytes, not the response object itself
    return Response(rsp.content, mimetype='audio/mpeg')


import io, json
import tempfile, os, io
from flask import jsonify

@app.route('/transcribe', methods=['POST'])
def transcribe():
    fs = request.files.get('file')
    if not fs:
        return jsonify(error="no file"), 400

    # save to temp
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        fs.save(tmp.name)
        tmp_path = tmp.name

    try:
        # decode via pydub→WAV
        audio = AudioSegment.from_file(tmp_path, format="webm")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.name = "speech.wav"
        wav_io.seek(0)

        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_io
        )
    finally:
        os.unlink(tmp_path)

    # ← access .text, not indexing
    return jsonify(text=resp.text)



@app.route('/finish/<node_id>', methods=['POST'])
def finish(node_id):
    node = student_ai.get_node(node_id)
    if not node:
        return "Knowledge node not found", 404
    # Aggregate all teacher messages from the conversation to update the knowledge base.
    aggregated_text = "\n".join([msg['message'] for msg in node.conversation if msg['role'] == 'user'])
    node.content = aggregated_text
    quiz = student_ai.generate_quiz(node)
    node.generated_quiz = quiz
    quiz_report = student_ai.auto_evaluate(node)
    node.quiz_report = quiz_report
    node._weak_dims = [d['zh_dim'] for d in quiz_report['details'] if not d['correct']]
    node.recap = None                     # ensure attribute exists
    return redirect(url_for('result', node_id=node.id))

@app.route('/recap/<node_id>')
def recap(node_id):
    node = student_ai.get_node(node_id)
    if not node:
        return "Node not found", 404
    if node.recap is None:                # first request → build it
        node.recap = student_ai.generate_recap(node, getattr(node, '_weak_dims', []))
    return jsonify(recap=node.recap or "All dimensions were correct – no recap needed!")

@app.route('/map_data')
def map_data():
    """
    Return all knowledge nodes + edges as JSON that D3 can digest.
    """
    nodes = []
    links = []
    # simple palette for first 10 categories
    cat_colors = {}
    palette = ["#3b82f6","#ef4444","#10b981","#f59e0b","#6366f1",
               "#ec4899","#8b5cf6","#22d3ee","#f97316","#14b8a6"]
    for idx,(nid,node) in enumerate(student_ai.knowledge_base.items()):
        if node.category not in cat_colors:
            cat_colors[node.category] = palette[len(cat_colors)%len(palette)]
        nodes.append({
            "id": nid,
            "label": node.title,
            "weight": node.weight,
            "category": node.category,
            "color": cat_colors[node.category]
        })
        # edges come from user-supplied related list
        for r in node.related:
            key = r.strip("#").lower().strip()
            target_id = student_ai.title_index.get(key) or key          # fall back to raw string
            if target_id in student_ai.knowledge_base:
                links.append({"source": nid, "target": target_id})
    return jsonify({"nodes": nodes, "links": links})

@app.route('/summary/<node_id>')
def summary(node_id):
    fmt = request.args.get('fmt','md')
    node = student_ai.get_node(node_id)
    if not node: return "Node not found",404

    md = f"""# Curistro Session – {node.title}

**Category**: {node.category}  
**Score**: {node.quiz_report['correct']} / {node.quiz_report['total']}

## Learner Explanation
{node.content}

## Clarifying Q & A
"""
    for m in node.conversation:
        role = "Teacher" if m['role']=='user' else "AI student"
        md += f"* **{role}:** {m['message']}\n"

    md += "\n## Quiz Questions & Answers\n"
    for q in node.generated_quiz['questions']:
        opts = "\n".join(f"  * {o}" for o in q['options'])
        md += f"### {q['zh_dim']}: {q['question']}\n{opts}\n*Correct:* {q['answer']}\n\n"

    if node.recap:
        md += f"## Adaptive Recap\n{node.recap}\n"

    if fmt=='md':
        resp = make_response(md)
        resp.headers['Content-Disposition'] = f'attachment; filename={node.id}.md'
        resp.mimetype = 'text/markdown'
        return resp

    # ---- PDF (requires WeasyPrint) ----
    if fmt=='pdf':
        #WKHTML = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"  # no quotes
        #config  = pdfkit.configuration(wkhtmltopdf=WKHTML)

        #html = markdown.markdown(md, extensions=["fenced_code"])
        #pdf  = pdfkit.from_string(
               # html, False,
              #  configuration=config,
             #   options={"quiet": ""}          # suppress progress
            #)

        #return send_file(io.BytesIO(pdf),
         #               download_name=f"{node.id}.pdf",
          #              as_attachment=True,
           #             mimetype="application/pdf")
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        # register DejaVu for full Unicode support
        pdf.add_font("DejaVu", "", os.path.join("static","fonts","DejaVuSans.ttf"), uni=True)
        pdf.set_font("DejaVu", size=12)

        for line in md.split("\n"):
            # reset to left margin so width=0 covers full content width
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 8, line)

        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return send_file(
            buf,
            download_name=f"{node.id}.pdf",
            as_attachment=True,
            mimetype="application/pdf"
        )



@app.route('/result/<node_id>')
def result(node_id):
    node = student_ai.get_node(node_id)
    if not node:
        return "Knowledge node not found", 404
    return render_template('result.html', node=node)

if __name__ == '__main__':
    app.run(debug=True)
