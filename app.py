import os
import hmac
import json
from hashlib import sha256
from flask import Flask, request, Response, send_file, abort
import threading
from pyvis.network import Network
import dropbox
from openai import OpenAI

# Configure logging before creating the Flask app
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
})

# Create necessary directories
os.makedirs("downloads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# --- CONFIGURATION ---
DROPBOX_APP_SECRET = os.environ.get("DROPBOX_APP_SECRET")

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- DROPBOX WEBHOOK ENDPOINTS ---

@app.route('/webhook', methods=['GET'])
def verify_dropbox():
    """Handle Dropbox webhook verification"""
    app.logger.info("Received Dropbox webhook verification request")
    challenge = request.args.get('challenge')
    resp = Response(challenge)
    resp.headers['Content-Type'] = 'text/plain'
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    return resp

@app.route('/webhook', methods=['POST'])
def dropbox_webhook():
    """Handle Dropbox file change notifications"""
    # Verify signature
    signature = request.headers.get('X-Dropbox-Signature')
    if not hmac.compare_digest(
        signature,
        hmac.new(DROPBOX_APP_SECRET.encode(), request.data, sha256).hexdigest()
    ):
        app.logger.error("Invalid Dropbox webhook signature")
        abort(403)
    
    data = json.loads(request.data)
    accounts = data.get('list_folder', {}).get('accounts', [])
    
    app.logger.info(f"Received webhook notification for {len(accounts)} account(s)")
    
    for account_id in accounts:
        app.logger.info(f"Starting processing for account: {account_id}")
        threading.Thread(target=process_dropbox_account, args=(account_id,)).start()
    
    return ''

def process_dropbox_account(account_id):
    """Process files for a specific Dropbox account"""
    try:
        app.logger.info(f"Processing started for account: {account_id}")
        
        # Download audio file
        app.logger.debug(f"Downloading latest audio file for account: {account_id}")
        audio_file_path = download_latest_audio_file(account_id)
        if not audio_file_path:
            app.logger.warning(f"No audio files found for account: {account_id}")
            return
        
        app.logger.info(f"Successfully downloaded audio file: {audio_file_path}")
        
        # Transcribe audio
        app.logger.debug("Starting audio transcription")
        transcript = transcribe_audio(audio_file_path)
        app.logger.info("Audio transcription completed")
        
        # Process with GPT-4
        app.logger.debug("Starting GPT-4 processing")
        summary = gpt4_process(transcript)
        app.logger.info("GPT-4 processing completed")
        
        # Generate mind map
        app.logger.debug("Generating mind map")
        mind_map_html = generate_mind_map(summary)
        
        # Save mind map
        mindmap_path = f"static/mindmap_{account_id}.html"
        with open(mindmap_path, "w") as f:
            f.write(mind_map_html)
        
        # Log completion and URL
        mindmap_url = f"https://{app.config.get('SERVER_NAME')}/mindmap/{account_id}"
        app.logger.info(f"Mind map successfully created for account: {account_id}")
        app.logger.info(f"Mind map URL: {mindmap_url} (CLICKABLE)")
        
    except Exception as e:
        app.logger.error(f"Error processing account {account_id}: {str(e)}", exc_info=True)

def download_latest_audio_file(account_id):
    """Download the latest audio file from Dropbox App Folder"""
    try:
        dbx = dropbox.Dropbox(os.environ.get("DROPBOX_ACCESS_TOKEN"))
        
        # Use the root of the App Folder
        folder_path = ""
        
        app.logger.debug(f"Listing files in Dropbox App Folder: {folder_path}")
        result = dbx.files_list_folder(folder_path)
        
        if not result.entries:
            app.logger.warning("No files found in App Folder")
            return None
        
        # Find the latest audio file (simplified example)
        # You might want to filter by file extension or other criteria
        latest_file = max(result.entries, key=lambda entry: entry.server_modified)
        
        local_path = f"downloads/{latest_file.name}"
        
        app.logger.debug(f"Downloading file: {latest_file.path_display}")
        dbx.files_download_to_file(local_path, latest_file.path_display)
        
        return local_path
    
    except dropbox.exceptions.ApiError as e:
        app.logger.error(f"Dropbox API error: {str(e)}")
        return None
    except Exception as e:
        app.logger.error(f"Error downloading file: {str(e)}")
        return None

# --- PROCESSING FUNCTIONS ---

def transcribe_audio(audio_file_path):
    """Transcribe audio using Whisper API"""
    with open(audio_file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=f
        )
    return transcript.text

def gpt4_process(transcript):
    """Process transcript with GPT-4"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Convert this transcript into mind map nodes:"},
            {"role": "user", "content": transcript}
        ]
    )
    return response.choices[0].message.content

def generate_mind_map(summary_text):
    """Generate Pyvis mind map from summary"""
    net = Network(height="750px", width="100%", directed=True)
    net.add_node("Summary", label="Summary")
    net.add_node("Details", label=summary_text[:40])
    net.add_edge("Summary", "Details")
    net.save_graph("templates/mindmap.html")
    with open("templates/mindmap.html") as f:
        return f.read()

# --- ROUTES ---

@app.route("/")
def index():
    """Main endpoint"""
    return "Dropbox-Whisper-GPT4-MindMap pipeline is running."

@app.route("/mindmap/<account_id>")
def view_mindmap(account_id):
    """Serve generated mind map"""
    path = f"static/mindmap_{account_id}.html"
    if not os.path.exists(path):
        app.logger.warning(f"Mind map not found for account: {account_id}")
        return f"No mind map found for account {account_id}", 404
    return send_file(path)

# --- MAIN ---

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
