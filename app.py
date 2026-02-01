"""
Audio Visualizer - 480p MP4 output for PowerPoint
Uses real ffmpeg for reliable H.264 encoding
"""

import os
import uuid
import numpy as np
import tempfile
import subprocess
import wave
import struct
from flask import Flask, render_template_string, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

UPLOAD_FOLDER = '/tmp/visualizer_uploads'
OUTPUT_FOLDER = '/tmp/visualizer_outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_audio_data(audio_path):
    """Extract audio data using ffmpeg"""
    temp_wav = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4().hex}.wav")
    
    cmd = [
        'ffmpeg', '-y', '-i', audio_path,
        '-ac', '1', '-ar', '44100', '-f', 'wav', temp_wav
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    
    with wave.open(temp_wav, 'rb') as wav:
        n_frames = wav.getnframes()
        sample_rate = wav.getframerate()
        raw_data = wav.readframes(n_frames)
        
    os.unlink(temp_wav)
    
    samples = np.array(struct.unpack(f'{n_frames}h', raw_data), dtype=np.float32)
    samples = samples / 32768.0
    
    return samples, sample_rate


def analyze_audio(samples, sample_rate, fps, num_bars):
    """Pre-compute frequency data for each frame"""
    duration = len(samples) / sample_rate
    total_frames = int(duration * fps)
    samples_per_frame = int(sample_rate / fps)
    
    frame_data = []
    
    for frame in range(total_frames):
        start = frame * samples_per_frame
        end = min(start + 2048, len(samples))
        chunk = samples[start:end]
        
        if len(chunk) < 64:
            frame_data.append([0.0] * num_bars)
            continue
        
        bands = []
        band_size = max(1, len(chunk) // num_bars)
        
        for i in range(num_bars):
            start_idx = i * band_size
            end_idx = min(start_idx + band_size, len(chunk))
            band_samples = chunk[start_idx:end_idx]
            energy = np.sqrt(np.mean(band_samples ** 2)) if len(band_samples) > 0 else 0
            freq_weight = 1 - (i / num_bars) * 0.4
            bands.append(min(1.0, energy * freq_weight * 8))
        
        if frame_data:
            prev = frame_data[-1]
            bands = [prev[i] * 0.3 + bands[i] * 0.7 for i in range(num_bars)]
        
        frame_data.append(bands)
    
    return frame_data, duration


def hsl_to_rgb(h, s, l):
    s /= 100
    l /= 100
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    
    if h < 60: r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else: r, g, b = c, 0, x
    
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


def get_color(color_name, index, total, value):
    colors = {
        'white': (255, 255, 255),
        'cyan': (0, 229, 255),
        'magenta': (255, 0, 204),
        'green': (51, 255, 102),
        'orange': (255, 128, 0),
        'gold': (255, 215, 0),
    }
    
    if color_name == 'cool':
        t = index / total
        return (int(t * 50), int(100 + t * 155), int(200 + t * 55))
    elif color_name == 'warm':
        t = index / total
        return (255, int(75 + t * 125), int(t * 50))
    elif color_name == 'rainbow':
        hue = (index / total) * 360
        return hsl_to_rgb(hue, 100, 50)
    
    return colors.get(color_name, (255, 255, 255))


def render_frame_to_raw(frame_data, num_bars, width, height, style, color, bar_height_mult):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :] = [10, 10, 11]
    
    center_y = height // 2
    
    if style == 'bars':
        total_width = int(width * 0.85)
        gap_ratio = 0.3
        bar_width = total_width / (num_bars + (num_bars - 1) * gap_ratio)
        gap_width = bar_width * gap_ratio
        start_x = (width - total_width) // 2
        max_height = int(height * 0.95 * bar_height_mult)
        
        for i, value in enumerate(frame_data):
            x = int(start_x + i * (bar_width + gap_width))
            bar_h = int(value * max_height)
            y_start = center_y - bar_h // 2
            y_end = center_y + bar_h // 2
            
            r, g, b = get_color(color, i, num_bars, value)
            
            x_end = min(int(x + bar_width), width)
            y_start = max(0, y_start)
            y_end = min(height, y_end)
            pixels[y_start:y_end, x:x_end] = [r, g, b]
    
    elif style == 'mirrored':
        total_width = int(width * 0.85)
        gap_ratio = 0.3
        bar_width = total_width / (num_bars + (num_bars - 1) * gap_ratio)
        gap_width = bar_width * gap_ratio
        start_x = (width - total_width) // 2
        max_height = int(height * 0.48 * bar_height_mult)
        
        for i, value in enumerate(frame_data):
            x = int(start_x + i * (bar_width + gap_width))
            bar_h = int(value * max_height)
            
            r, g, b = get_color(color, i, num_bars, value)
            x_end = min(int(x + bar_width), width)
            
            pixels[center_y:min(center_y + bar_h, height), x:x_end] = [r, g, b]
            pixels[max(center_y - bar_h, 0):center_y, x:x_end] = [r, g, b]
    
    elif style == 'circular':
        center_x = width // 2
        inner_radius = min(width, height) * 0.12
        max_bar_length = min(width, height) * 0.35 * bar_height_mult
        
        for i, value in enumerate(frame_data):
            angle = (i / num_bars) * 2 * np.pi - np.pi / 2
            bar_length = inner_radius + value * max_bar_length
            
            r, g, b = get_color(color, i, num_bars, value)
            
            for t in np.linspace(0, 1, int(bar_length - inner_radius) + 1):
                radius = inner_radius + t * (bar_length - inner_radius)
                px = int(center_x + radius * np.cos(angle))
                py = int(center_y + radius * np.sin(angle))
                if 0 <= px < width and 0 <= py < height:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if 0 <= px+dx < width and 0 <= py+dy < height:
                                pixels[py+dy, px+dx] = [r, g, b]
    
    elif style == 'waveform':
        amplitude = int(height * 0.45 * bar_height_mult)
        start_x = int(width * 0.05)
        end_x = int(width * 0.95)
        step = (end_x - start_x) / (num_bars - 1)
        
        r, g, b = get_color(color, 0, num_bars, 0.5)
        
        for i, value in enumerate(frame_data):
            x = int(start_x + i * step)
            y_offset = int(value * amplitude)
            
            y_top = center_y - y_offset
            y_bottom = center_y + y_offset
            
            for y in range(max(0, y_top), min(height, y_bottom)):
                if 0 <= x < width:
                    pixels[y, x] = [r, g, b]
                    if x + 1 < width:
                        pixels[y, x + 1] = [r, g, b]
    
    return pixels.tobytes()


def create_video(audio_path, frame_data, duration, output_path, fps, num_bars, style, color, bar_height_mult):
    width, height = 854, 480
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-i', audio_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        '-movflags', '+faststart',
        output_path
    ]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    total_frames = len(frame_data)
    
    for i, bands in enumerate(frame_data):
        frame_bytes = render_frame_to_raw(bands, num_bars, width, height, style, color, bar_height_mult)
        process.stdin.write(frame_bytes)
        
        if (i + 1) % 100 == 0:
            print(f"Frame {i + 1}/{total_frames}")
    
    process.stdin.close()
    process.wait()
    
    return output_path


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #0a0a0b; color: #fff; min-height: 100vh; padding: 40px 20px; }
        .container { max-width: 700px; margin: 0 auto; }
        h1 { text-align: center; font-weight: 300; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .badge { background: #1a1a1a; color: #4ade80; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; }
        
        .drop-zone { border: 2px dashed #333; border-radius: 16px; padding: 50px; text-align: center; cursor: pointer; background: #111; margin-bottom: 25px; transition: all 0.3s; }
        .drop-zone:hover, .drop-zone.dragover { border-color: #fff; background: #1a1a1a; }
        .drop-zone.has-file { border-color: #4ade80; }
        .drop-zone-icon { font-size: 2.5rem; margin-bottom: 10px; opacity: 0.5; }
        .file-name { color: #4ade80; font-weight: 500; }
        
        .settings { background: #111; border-radius: 16px; padding: 25px; margin-bottom: 25px; }
        .settings h2 { font-size: 1rem; color: #888; margin-bottom: 20px; }
        .setting-group { margin-bottom: 20px; }
        .setting-label { display: block; font-size: 0.85rem; color: #666; margin-bottom: 8px; text-transform: uppercase; }
        
        .options { display: flex; gap: 8px; flex-wrap: wrap; }
        .option-btn { padding: 10px 16px; border: 1px solid #333; background: transparent; color: #fff; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
        .option-btn:hover { border-color: #555; }
        .option-btn.active { border-color: #fff; background: #fff; color: #000; }
        
        .color-btn { width: 36px; height: 36px; border-radius: 50%; border: 2px solid #333; cursor: pointer; transition: transform 0.2s; }
        .color-btn:hover { transform: scale(1.1); }
        .color-btn.active { border-color: #fff; box-shadow: 0 0 0 2px #0a0a0b, 0 0 0 4px #fff; }
        
        .slider-row { display: flex; align-items: center; gap: 12px; }
        .slider { flex: 1; -webkit-appearance: none; height: 5px; background: #333; border-radius: 3px; }
        .slider::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; background: #fff; border-radius: 50%; cursor: pointer; }
        .slider-value { min-width: 35px; text-align: right; color: #888; font-size: 0.9rem; }
        
        .generate-btn { width: 100%; padding: 16px; font-size: 1rem; font-weight: 500; background: #fff; color: #000; border: none; border-radius: 10px; cursor: pointer; transition: all 0.2s; }
        .generate-btn:hover:not(:disabled) { background: #eee; }
        .generate-btn:disabled { opacity: 0.3; cursor: not-allowed; }
        
        .progress { display: none; background: #111; border-radius: 16px; padding: 25px; margin-bottom: 25px; }
        .progress.visible { display: block; }
        .progress-bar-bg { background: #222; border-radius: 8px; height: 16px; overflow: hidden; margin-bottom: 12px; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #4ade80, #22c55e); width: 0%; transition: width 0.3s; }
        .progress-text { text-align: center; color: #888; }
        
        .result { display: none; background: #111; border-radius: 16px; padding: 25px; text-align: center; }
        .result.visible { display: block; }
        .result video { width: 100%; border-radius: 10px; margin-bottom: 20px; background: #000; }
        .download-btn { display: inline-block; padding: 12px 35px; font-size: 1rem; background: #4ade80; color: #000; border: none; border-radius: 8px; cursor: pointer; text-decoration: none; }
        .download-btn:hover { background: #22c55e; }
        .new-btn { padding: 12px 35px; background: transparent; color: #888; border: 1px solid #333; border-radius: 8px; cursor: pointer; margin-left: 10px; }
        
        footer { text-align: center; margin-top: 40px; color: #444; font-size: 0.8rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Audio Visualizer <span class="badge">480p MP4</span></h1>
        <p class="subtitle">Create PowerPoint-compatible visualizations from audio</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="fileInput" name="audio" accept="audio/*" hidden>
            <div class="drop-zone" id="dropZone">
                <div class="drop-zone-icon">🎵</div>
                <div id="dropText">Drop audio file here or click to browse</div>
            </div>
            
            <div class="settings">
                <h2>Settings</h2>
                
                <div class="setting-group">
                    <label class="setting-label">Style</label>
                    <div class="options" id="styleOptions">
                        <button type="button" class="option-btn active" data-value="bars">Bars</button>
                        <button type="button" class="option-btn" data-value="mirrored">Mirrored</button>
                        <button type="button" class="option-btn" data-value="circular">Circular</button>
                        <button type="button" class="option-btn" data-value="waveform">Waveform</button>
                    </div>
                    <input type="hidden" name="style" id="styleInput" value="bars">
                </div>
                
                <div class="setting-group">
                    <label class="setting-label">Color</label>
                    <div class="options" id="colorOptions">
                        <button type="button" class="color-btn active" data-value="white" style="background:#fff"></button>
                        <button type="button" class="color-btn" data-value="cyan" style="background:#00e5ff"></button>
                        <button type="button" class="color-btn" data-value="magenta" style="background:#ff00cc"></button>
                        <button type="button" class="color-btn" data-value="green" style="background:#33ff66"></button>
                        <button type="button" class="color-btn" data-value="orange" style="background:#ff8000"></button>
                        <button type="button" class="color-btn" data-value="gold" style="background:#ffd700"></button>
                        <button type="button" class="color-btn" data-value="cool" style="background:linear-gradient(135deg,#0066ff,#00ffff)"></button>
                        <button type="button" class="color-btn" data-value="warm" style="background:linear-gradient(135deg,#ff6600,#ffcc00)"></button>
                        <button type="button" class="color-btn" data-value="rainbow" style="background:linear-gradient(90deg,#ff0000,#ff8000,#ffff00,#00ff00,#0080ff,#8000ff)"></button>
                    </div>
                    <input type="hidden" name="color" id="colorInput" value="white">
                </div>
                
                <div class="setting-group">
                    <label class="setting-label">Number of Bars</label>
                    <div class="slider-row">
                        <input type="range" class="slider" name="num_bars" id="numBars" min="16" max="128" value="64">
                        <span class="slider-value" id="numBarsVal">64</span>
                    </div>
                </div>
                
                <div class="setting-group">
                    <label class="setting-label">Bar Height</label>
                    <div class="slider-row">
                        <input type="range" class="slider" name="bar_height" id="barHeight" min="30" max="100" value="80">
                        <span class="slider-value" id="barHeightVal">80%</span>
                    </div>
                </div>
            </div>
            
            <button type="submit" class="generate-btn" id="generateBtn" disabled>Generate Video</button>
        </form>
        
        <div class="progress" id="progressSection">
            <div class="progress-bar-bg"><div class="progress-bar" id="progressBar"></div></div>
            <div class="progress-text" id="progressText">Processing...</div>
        </div>
        
        <div class="result" id="resultSection">
            <video id="resultVideo" controls></video>
            <div>
                <a class="download-btn" id="downloadBtn" href="#" download="visualizer.mp4">Download MP4</a>
                <button class="new-btn" id="newBtn">Create Another</button>
            </div>
        </div>
        
        <footer>854×480 • 30fps • H.264 MP4 • PowerPoint Compatible</footer>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const dropText = document.getElementById('dropText');
        const generateBtn = document.getElementById('generateBtn');
        const uploadForm = document.getElementById('uploadForm');
        const progressSection = document.getElementById('progressSection');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const resultSection = document.getElementById('resultSection');
        const resultVideo = document.getElementById('resultVideo');
        const downloadBtn = document.getElementById('downloadBtn');
        const newBtn = document.getElementById('newBtn');
        
        dropZone.onclick = () => fileInput.click();
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
        dropZone.ondragleave = () => dropZone.classList.remove('dragover');
        dropZone.ondrop = (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); if(e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; updateFile(); } };
        fileInput.onchange = updateFile;
        
        function updateFile() {
            if (fileInput.files[0]) {
                dropZone.classList.add('has-file');
                dropText.innerHTML = '<div class="file-name">' + fileInput.files[0].name + '</div><small style="color:#666">Click to change</small>';
                generateBtn.disabled = false;
            }
        }
        
        document.querySelectorAll('#styleOptions .option-btn').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('#styleOptions .option-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('styleInput').value = btn.dataset.value;
            };
        });
        
        document.querySelectorAll('#colorOptions .color-btn').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('#colorOptions .color-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('colorInput').value = btn.dataset.value;
            };
        });
        
        document.getElementById('numBars').oninput = function() { document.getElementById('numBarsVal').textContent = this.value; };
        document.getElementById('barHeight').oninput = function() { document.getElementById('barHeightVal').textContent = this.value + '%'; };
        
        uploadForm.onsubmit = async (e) => {
            e.preventDefault();
            
            uploadForm.style.display = 'none';
            progressSection.classList.add('visible');
            progressBar.style.width = '10%';
            progressText.textContent = 'Uploading audio...';
            
            const formData = new FormData(uploadForm);
            
            try {
                progressText.textContent = 'Processing audio and generating video... This may take a minute.';
                progressBar.style.width = '30%';
                
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });
                
                progressBar.style.width = '90%';
                
                const result = await response.json();
                
                if (result.error) {
                    alert('Error: ' + result.error);
                    uploadForm.style.display = 'block';
                    progressSection.classList.remove('visible');
                    return;
                }
                
                progressBar.style.width = '100%';
                progressText.textContent = 'Complete!';
                
                resultVideo.src = '/download/' + result.video_id + '?t=' + Date.now();
                downloadBtn.href = '/download/' + result.video_id;
                downloadBtn.download = 'visualizer_' + Date.now() + '.mp4';
                
                progressSection.classList.remove('visible');
                resultSection.classList.add('visible');
                
            } catch (err) {
                alert('Error: ' + err.message);
                uploadForm.style.display = 'block';
                progressSection.classList.remove('visible');
            }
        };
        
        newBtn.onclick = () => {
            resultSection.classList.remove('visible');
            uploadForm.style.display = 'block';
            fileInput.value = '';
            dropZone.classList.remove('has-file');
            dropText.innerHTML = 'Drop audio file here or click to browse';
            generateBtn.disabled = true;
            resultVideo.src = '';
        };
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    style = request.form.get('style', 'bars')
    color = request.form.get('color', 'white')
    num_bars = int(request.form.get('num_bars', 64))
    bar_height = int(request.form.get('bar_height', 80)) / 100
    
    job_id = uuid.uuid4().hex[:8]
    filename = secure_filename(file.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
    file.save(audio_path)
    
    try:
        print(f"Processing: {filename}")
        
        samples, sample_rate = get_audio_data(audio_path)
        frame_data, duration = analyze_audio(samples, sample_rate, 30, num_bars)
        
        print(f"Duration: {duration:.1f}s, Frames: {len(frame_data)}")
        
        output_filename = f"visualizer_{job_id}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        create_video(audio_path, frame_data, duration, output_path, 30, num_bars, style, color, bar_height)
        
        os.unlink(audio_path)
        
        print(f"Done: {output_filename}")
        
        return jsonify({
            'success': True,
            'video_id': output_filename,
            'duration': round(duration, 2)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        return jsonify({'error': str(e)}), 500


@app.route('/download/<video_id>')
def download(video_id):
    video_path = os.path.join(OUTPUT_FOLDER, secure_filename(video_id))
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    return send_file(video_path, mimetype='video/mp4')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
