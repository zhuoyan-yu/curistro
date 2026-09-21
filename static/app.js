'use strict';

const csrf = document.querySelector('meta[name="csrf-token"]').content;
const errorBox = document.getElementById('error');
const statusBox = document.getElementById('status');
const sessionBox = document.getElementById('session');
const nodeId = sessionBox?.dataset.nodeId;
let running = false;

function controls(disabled) {
  if (disabled) {
    const previous = [...document.querySelectorAll('button')].map(button => [button, button.disabled]);
    previous.forEach(([button]) => { button.disabled = true; });
    return () => previous.forEach(([button, value]) => { button.disabled = value; });
  }
}

async function busy(label, operation) {
  if (running) return;
  running = true;
  const restore = controls(true);
  errorBox.hidden = true;
  statusBox.textContent = label;
  try { await operation(); }
  catch (error) {
    errorBox.textContent = error.message || 'The request could not finish. Please retry.';
    errorBox.hidden = false;
    errorBox.scrollIntoView({block: 'nearest'});
  } finally {
    restore(); running = false; statusBox.textContent = '';
  }
}

async function request(url, options = {}) {
  const response = await fetch(url, {...options, headers: {
    'Accept': 'application/json', 'X-CSRF-Token': csrf, ...options.headers
  }});
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || 'The request failed (' + response.status + '). Please retry.');
  }
  return response;
}

async function post(url, data = {}) {
  return (await request(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)})).json();
}

const teachForm = document.getElementById('teachForm');
teachForm?.addEventListener('submit', event => {
  event.preventDefault();
  busy('Starting your teaching session…', async () => {
    const response = await request(teachForm.action, {method: 'POST', body: new FormData(teachForm)});
    location.assign(response.url);
  });
});

document.getElementById('exampleBtn')?.addEventListener('click', () => {
  document.getElementById('title').value = 'Photosynthesis';
  document.getElementById('category').value = 'Science';
  document.getElementById('content').value = 'Plants use light energy, water, and carbon dioxide to make sugars and release oxygen. Chlorophyll absorbs light. The sugars store chemical energy that the plant can use later.';
  document.getElementById('content').focus();
});

const conversation = document.getElementById('conversation');
function bubble(role, text) {
  const article = document.createElement('article');
  article.className = 'message ' + role;
  const label = document.createElement('span');
  label.className = 'speaker'; label.textContent = role === 'user' ? 'You · teacher' : 'AI student';
  const content = document.createElement('div');
  content.className = 'message-text'; content.textContent = text;
  article.append(label, content); conversation.append(article);
  return {article, content};
}

async function send(message) {
  const response = await request('/reply/' + nodeId, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message})
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', answer = null, completed = false;
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      let split;
      while ((split = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, split); buffer = buffer.slice(split + 2);
        const lines = frame.split('\n');
        const kind = lines.find(line => line.startsWith('event:'))?.slice(6).trim();
        const payload = JSON.parse(lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n'));
        if (kind === 'accepted' && payload.message) {
          bubble('user', payload.message); document.getElementById('message').value = '';
        }
        if (kind === 'token') {
          answer ||= bubble('ai', '');
          answer.content.textContent += payload.text;
        }
        if (kind === 'error') throw new Error(payload.message);
        if (kind === 'done') completed = true;
      }
      if (done) break;
    }
    if (!completed) throw new Error('The connection ended before the reply finished. Use Retry AI question.');
  } catch (error) {
    answer?.article.remove();
    throw error;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

document.getElementById('msgForm')?.addEventListener('submit', event => {
  event.preventDefault();
  const message = document.getElementById('message').value.trim();
  if (message) busy('Your AI student is thinking…', () => send(message));
});
document.getElementById('retryBtn')?.addEventListener('click', () => busy('Retrying the AI question…', () => send('')));
document.getElementById('explainBtn')?.addEventListener('click', () => busy('Preparing an explain-back…', async () => {
  const data = await post('/explain/' + nodeId); bubble('ai', data.text);
}));
document.getElementById('finishBtn')?.addEventListener('click', () => busy('Generating four questions and checking the AI student…', async () => {
  const data = await post('/finish/' + nodeId); location.assign(data.url);
}));
document.getElementById('recapBtn')?.addEventListener('click', () => busy('Preparing a follow-up recap…', async () => {
  const data = await post('/recap/' + nodeId);
  document.getElementById('recapText').textContent = data.recap;
  document.getElementById('recapBtn').hidden = true;
}));

document.getElementById('ttsBtn')?.addEventListener('click', () => busy('Preparing AI-generated speech…', async () => {
  const response = await request('/speak/' + nodeId, {method: 'POST'});
  const url = URL.createObjectURL(await response.blob());
  const audio = new Audio(url);
  audio.onended = audio.onerror = () => URL.revokeObjectURL(url);
  try { await audio.play(); } catch (error) { URL.revokeObjectURL(url); throw error; }
}));

let recorder, mediaStream, recordTimer, leaving = false;
const recordButton = document.getElementById('recBtn');
recordButton?.addEventListener('click', async () => {
  if (recorder?.state === 'recording') { recorder.stop(); return; }
  if (running) return;
  running = true;
  const restore = controls(true);
  errorBox.hidden = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error('This browser cannot record audio. Use text input or a browser with recording support.');
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mime) throw new Error('No supported recording format is available in this browser.');
    mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});
    recorder = new MediaRecorder(mediaStream, {mimeType: mime});
    const chunks = [];
    let recordingFailed = false;
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onerror = () => { recordingFailed = true; if (recorder.state !== 'inactive') recorder.stop(); };
    recorder.onstop = () => {
      clearTimeout(recordTimer); mediaStream.getTracks().forEach(track => track.stop());
      restore(); running = false; recordButton.textContent = 'Record explanation'; statusBox.textContent = '';
      if (leaving) return;
      busy('Transcribing your recording…', async () => {
        if (recordingFailed) throw new Error('Recording failed. Please try again.');
        const blob = new Blob(chunks, {type: mime});
        if (!blob.size) throw new Error('The recording is empty. Please try again.');
        if (blob.size > 9 * 1024 * 1024) throw new Error('The recording is too large. Please record a shorter explanation.');
        const form = new FormData();
        form.append('file', blob, mime.includes('mp4') ? 'recording.mp4' : 'recording.webm');
        const data = await (await request('/transcribe/' + nodeId, {method: 'POST', body: form})).json();
        document.getElementById('message').value = data.text;
      });
    };
    recorder.start(); recordButton.disabled = false; recordButton.textContent = 'Stop recording';
    statusBox.textContent = 'Recording… stops automatically after 60 seconds.';
    recordTimer = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 60000);
  } catch (error) {
    mediaStream?.getTracks().forEach(track => track.stop());
    restore(); running = false; errorBox.textContent = error.message; errorBox.hidden = false;
  }
});
window.addEventListener('pagehide', () => {
  leaving = true; clearTimeout(recordTimer);
  if (recorder?.state === 'recording') recorder.stop();
  mediaStream?.getTracks().forEach(track => track.stop());
});

let mapSimulation;
document.getElementById('mapBtn')?.addEventListener('click', () => busy('Drawing your topic map…', async () => {
  const {nodes, links} = await (await request('/map_data')).json();
  const list = document.getElementById('mapList'); list.replaceChildren();
  const labels = new Map(nodes.map(node => [node.id, node.label]));
  const descriptions = links.length ? links.map(link => labels.get(link.source) + ' → ' + labels.get(link.target)) : nodes.map(node => node.label + ' — no linked topic yet');
  descriptions.forEach(text => { const li = document.createElement('li'); li.textContent = text; list.append(li); });
  if (!window.d3) throw new Error('The graph library could not load. Your topic links are listed below.');
  mapSimulation?.stop();
  document.getElementById('mapArea').replaceChildren();
  const width = 760, height = 360;
  const svg = d3.select('#mapArea').append('svg').attr('viewBox', '0 0 ' + width + ' ' + height).attr('role', 'img').attr('aria-label', 'Connected teaching topics');
  const color = d3.scaleOrdinal(d3.schemeTableau10);
  const edges = svg.append('g').attr('stroke', '#a5b1c5').selectAll('line').data(links).join('line');
  const points = svg.append('g').selectAll('g').data(nodes).join('g');
  points.append('circle').attr('r', 10).attr('fill', node => color(node.category));
  points.append('title').text(node => node.label + ' · ' + node.category);
  points.append('text').text(node => node.label.length > 28 ? node.label.slice(0, 27) + '…' : node.label).attr('x', 14).attr('y', 5).attr('font-size', 12).attr('fill', '#19243a');
  mapSimulation = d3.forceSimulation(nodes).force('link', d3.forceLink(links).id(node => node.id).distance(150)).force('charge', d3.forceManyBody().strength(-250)).force('center', d3.forceCenter(width / 2, height / 2));
  points.call(d3.drag().on('start', (event, node) => { if (!event.active) mapSimulation.alphaTarget(.3).restart(); node.fx = node.x; node.fy = node.y; }).on('drag', (event, node) => { node.fx = event.x; node.fy = event.y; }).on('end', (event, node) => { if (!event.active) mapSimulation.alphaTarget(0); node.fx = node.fy = null; }));
  mapSimulation.on('tick', () => {
    nodes.forEach(node => { node.x = Math.max(20, Math.min(width - 190, node.x)); node.y = Math.max(20, Math.min(height - 20, node.y)); });
    edges.attr('x1', link => link.source.x).attr('y1', link => link.source.y).attr('x2', link => link.target.x).attr('y2', link => link.target.y);
    points.attr('transform', node => 'translate(' + node.x + ',' + node.y + ')');
  });
}));

document.querySelectorAll('[data-download]').forEach(button => button.addEventListener('click', () => busy('Preparing your export…', async () => {
  const response = await request(button.dataset.download);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a'); link.href = url;
  link.download = 'curistro-session.' + (button.dataset.download.endsWith('pdf') ? 'pdf' : 'md');
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
})));
