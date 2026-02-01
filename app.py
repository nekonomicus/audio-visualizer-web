"""
Audio Visualizer Web App
Generates MP4 videos with synced audio visualizations
"""

import os
import uuid
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tempfile
import subprocess
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Color presets
COLOR_PRESETS = {
    'white': {'bar': [1.0, 1.0, 1.0], 'bg': '#0a0a0b'},
    'cyan': {'bar': [0.0, 0.9, 1.0], 'bg': '#0a0a0b'},
    'magenta': {'bar': [1.0, 0.0, 0.8], 'bg': '#0a0a0b'},
    'green': {'bar': [0.2, 1.0, 0.4], 'bg': '#0a0a0b'},
    'orange': {'bar': [1.0, 0.5, 0.0], 'bg': '#0a0a0b'},
    'gold': {'bar': [1.0, 0.84, 0.0], 'bg': '#0a0a0b'},
    'cool_gradient': {'bar': 'cool', 'bg': '#0a0a0b'},
    'warm_gradient': {'bar': 'warm', 'bg': '#0a0a0b'},
}

# Visualization styles
VIZ_STYLES = ['bars', 'mirrored', 'circular', 'waveform']


def get_bar_color(preset, index, total, value):
    """Get color for a bar based on preset"""
    config = COLOR_PRESETS.get(preset, COLOR_PRESETS['white'])
    bar_config = config['bar']
    
    if bar_config == 'cool':
        # Blue to cyan gradient
        t = index / total
        r = 0.0 + t * 0.2
        g = 0.4 + t * 0.5
        b = 0.8 + t * 0.2
        return [r, g, b]
    elif bar_config == 'warm':
        # Orange to yellow gradient
        t = index / total
        r = 1.0
        g = 0.3 + t * 0.5
        b = 0.0 + t * 0.2
        return [r, g, b]
    else:
        return bar_config


def generate_visualizer_frames(audio_path, style='bars', color='white', num_bars=64, fps=30):
    """Generate visualization frames from audio file"""
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Calculate hop length to sync with video FPS
    hop_length = int(sr / fps)
    
    # Get mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_bars, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize to 0-1 range
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    
    # Apply some smoothing for cleaner visuals
    from scipy.ndimage import uniform_filter1d
    S_smooth = uniform_filter1d(S_norm, size=3, axis=1)
    
    total_frames = S_smooth.shape[1]
    bg_color = COLOR_PRESETS.get(color, COLOR_PRESETS['white'])['bg']
    
    frames = []
    
    for frame_idx in range(total_frames):
        audio_data = S_smooth[:, frame_idx]
        
        # Boost lower frequencies slightly for visual impact
        boost = np.linspace(1.2, 0.8, num_bars)
        audio_data = np.clip(audio_data * boost, 0, 1)
        
        frame = render_frame(audio_data, style, color, num_bars, bg_color)
        frames.append(frame)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Frame {frame_idx + 1}/{total_frames}")
    
    return frames, duration, fps


def render_frame(audio_data, style, color, num_bars, bg_color):
    """Render a single frame"""
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)  # 1920x1080
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    if style == 'bars':
        render_bars(ax, audio_data, color, num_bars, mirrored=False)
    elif style == 'mirrored':
        render_bars(ax, audio_data, color, num_bars, mirrored=True)
    elif style == 'circular':
        render_circular(ax, audio_data, color, num_bars, bg_color)
    elif style == 'waveform':
        render_waveform(ax, audio_data, color, num_bars)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_aspect('auto')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Convert to image array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return frame


def render_bars(ax, audio_data, color, num_bars, mirrored=False):
    """Render bar visualization"""
    total_width = 0.85
    gap_ratio = 0.3
    bar_width = total_width / (num_bars + (num_bars - 1) * gap_ratio)
    gap_width = bar_width * gap_ratio
    start_x = (1 - total_width) / 2
    
    for i, value in enumerate(audio_data):
        x = start_x + i * (bar_width + gap_width)
        bar_color = get_bar_color(color, i, num_bars, value)
        
        if mirrored:
            height = value * 0.4
            y = 0.5 - height
            # Top bars
            alpha = 0.7 + value * 0.3
            rect_top = Rectangle((x, 0.5), bar_width, height, 
                                  facecolor=(*bar_color, alpha), edgecolor='none')
            ax.add_patch(rect_top)
            # Bottom bars (mirrored)
            rect_bottom = Rectangle((x, 0.5 - height), bar_width, height,
                                     facecolor=(*bar_color, alpha), edgecolor='none')
            ax.add_patch(rect_bottom)
        else:
            height = value * 0.55
            y = 0.5 - height / 2
            alpha = 0.7 + value * 0.3
            rect = Rectangle((x, y), bar_width, height,
                              facecolor=(*bar_color, alpha), edgecolor='none')
            ax.add_patch(rect)
            
            # Reflection
            ref_height = height * 0.12
            reflection = Rectangle(
                (x, y - ref_height - 0.008), bar_width, ref_height,
                facecolor=(*bar_color, alpha * 0.12), edgecolor='none'
            )
            ax.add_patch(reflection)


def render_circular(ax, audio_data, color, num_bars, bg_color):
    """Render circular visualization"""
    center_x, center_y = 0.5, 0.5
    inner_radius = 0.15
    max_bar_length = 0.25
    
    for i, value in enumerate(audio_data):
        angle = (i / num_bars) * 2 * np.pi - np.pi / 2
        bar_color = get_bar_color(color, i, num_bars, value)
        alpha = 0.7 + value * 0.3
        
        bar_length = inner_radius + value * max_bar_length
        
        x1 = center_x + inner_radius * np.cos(angle)
        y1 = center_y + inner_radius * np.sin(angle)
        x2 = center_x + bar_length * np.cos(angle)
        y2 = center_y + bar_length * np.sin(angle)
        
        ax.plot([x1, x2], [y1, y2], color=(*bar_color, alpha), 
                linewidth=3, solid_capstyle='round')


def render_waveform(ax, audio_data, color, num_bars):
    """Render waveform visualization"""
    x = np.linspace(0.05, 0.95, num_bars)
    y_center = 0.5
    
    # Create smooth curve
    y_top = y_center + audio_data * 0.3
    y_bottom = y_center - audio_data * 0.3
    
    bar_color = get_bar_color(color, 0, num_bars, 0.5)
    
    # Fill between
    ax.fill_between(x, y_bottom, y_top, color=(*bar_color, 0.6), edgecolor='none')
    ax.plot(x, y_top, color=(*bar_color, 0.9), linewidth=2)
    ax.plot(x, y_bottom, color=(*bar_color, 0.9), linewidth=2)


def create_video_with_audio(frames, audio_path, output_path, fps=30):
    """Create video from frames and merge with audio using ffmpeg"""
    
    # Create temporary video without audio
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video_path = temp_video.name
    temp_video.close()
    
    # Write frames to temp video using imageio
    import imageio
    writer = imageio.get_writer(temp_video_path, fps=fps, codec='libx264', 
                                 quality=8, pixelformat='yuv420p')
    
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    # Merge video with audio using ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_video_path,
        '-i', audio_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Clean up temp file
    os.unlink(temp_video_path)
    
    return output_path


@app.route('/')
def index():
    return render_template('index.html', 
                           colors=list(COLOR_PRESETS.keys()),
                           styles=VIZ_STYLES)


@app.route('/generate', methods=['POST'])
def generate():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp3, wav, ogg, m4a, flac, aac'}), 400
    
    # Get options
    style = request.form.get('style', 'bars')
    color = request.form.get('color', 'white')
    num_bars = int(request.form.get('num_bars', 64))
    
    # Validate options
    if style not in VIZ_STYLES:
        style = 'bars'
    if color not in COLOR_PRESETS:
        color = 'white'
    num_bars = max(16, min(128, num_bars))
    
    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(audio_path)
    
    try:
        # Generate visualization
        print(f"Generating visualization for {filename}...")
        frames, duration, fps = generate_visualizer_frames(
            audio_path, style=style, color=color, num_bars=num_bars
        )
        
        # Create output video with audio
        output_filename = f"visualizer_{job_id}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        print(f"Creating video with audio...")
        create_video_with_audio(frames, audio_path, output_path, fps=fps)
        
        # Clean up uploaded file
        os.unlink(audio_path)
        
        return jsonify({
            'success': True,
            'video_id': output_filename,
            'duration': round(duration, 2)
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        return jsonify({'error': str(e)}), 500


@app.route('/download/<video_id>')
def download(video_id):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(video_id))
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    
    return send_file(video_path, as_attachment=True, download_name=video_id)


@app.route('/preview/<video_id>')
def preview(video_id):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(video_id))
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    
    return send_file(video_path, mimetype='video/mp4')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
