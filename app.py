import os
import hmac
import json
from hashlib import sha256
from flask import Flask, request, Response, render_template, jsonify, send_file, abort
import requests
import threading
from pyvis.network import Network
import dropbox
from openai import OpenAI


os.makedirs("downloads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# --- CONFIGURATION ---
DROPBOX_APP_SECRET = os.environ.get("DROPBOX_APP_SECRET")

app = Flask(__name__)

# --- DROPBOX WEBHOOK ENDPOINTS ---

@app.route('/webhook', methods=['GET'])
def verify_dropbox():
    # Respond to Dropbox webhook challenge
    challenge = request.args.get('challenge')
    resp = Response(challenge)
    resp.headers['Content-Type'] = 'text/plain'
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    return resp

@app.route('/webhook', methods=['POST'])
def dropbox_webhook():
    # Verify Dropbox signature
    signature = request.headers.get('X-Dropbox-Signature')
    if not hmac.compare_digest(
        signature,
        hmac.new(DROPBOX_APP_SECRET.encode(), request.data, sha256).hexdigest()
    ):
        abort(403)
    # Process each account in a separate thread
    for account in json.loads(request.data)['list_folder']['accounts']:
        threading.Thread(target=process_dropbox_account, args=(account,)).start()
    return ''

def process_dropbox_account(account_id):
    audio_file_path = download_latest_audio_file(account_id)
    if not audio_file_path:
        print(f"No audio file found for account {account_id}")
        return
    transcript = transcribe_audio(audio_file_path)
    summary = gpt4_process(transcript)
    mind_map_html = generate_mind_map(summary)
    with open(f"static/mindmap_{account_id}.html", "w") as f:
        f.write(mind_map_html)
    # Optionally: push to GitHub Pages using ghp-import or similar

def download_latest_audio_file(account_id):
    dbx = dropbox.Dropbox(os.environ.get("DROPBOX_ACCESS_TOKEN"))
    # List files in a specific folder (adjust as needed)
    folder_path = f"/account_{account_id}"
    result = dbx.files_list_folder(folder_path)
    if not result.entries:
        return None
    # Get the latest file (simplified)
    latest_file = result.entries[0]
    local_path = f"downloads/{latest_file.name}"
    dbx.files_download_to_file(local_path, latest_file.path_display)
    return local_path

# --- WHISPER TRANSCRIPTION ---

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=f
        )
    return transcript.text

# --- GPT-4 (or Free Alternative) PROCESSING ---

def gpt4_process(transcript):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Convert this transcript into mind map nodes:"},
            {"role": "user", "content": transcript}
        ]
    )
    return response.choices[0].message.content

# --- PYVIS MIND MAP GENERATION ---

def generate_mind_map(summary_text):
    # For demo: parse summary_text into nodes/edges (implement your own logic)
    net = Network(height="750px", width="100%", directed=True)
    net.add_node("Summary", label="Summary")
    net.add_node("Details", label=summary_text[:40])
    net.add_edge("Summary", "Details")
    # Save and return HTML
    net.save_graph("templates/mindmap.html")
    with open("templates/mindmap.html") as f:
        return f.read()

# --- FLASK ROUTES FOR VIEWING MIND MAPS ---

@app.route("/")
def index():
    return "Dropbox-Whisper-GPT4-MindMap pipeline is running."

@app.route("/mindmap/<account_id>")
def view_mindmap(account_id):
    path = f"static/mindmap_{account_id}.html"
    if not os.path.exists(path):
        return f"No mind map found for account {account_id}", 404
    return send_file(path)

# --- MAIN ---

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
