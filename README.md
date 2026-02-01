# Audio Visualizer Web App

Generate stunning MP4 visualizations from audio files with synced audio.

## Features

- **Drag & Drop**: Simply drop your audio file to get started
- **Multiple Styles**: Bars, Mirrored, Circular, Waveform
- **Color Options**: White, Cyan, Magenta, Green, Orange, Gold, Cool/Warm Gradients
- **Adjustable Bars**: 16-128 frequency bars
- **Full HD Output**: 1920×1080 at 30fps
- **Audio Included**: Output MP4 contains the original audio

## Supported Formats

MP3, WAV, OGG, M4A, FLAC, AAC

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure ffmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: apt-get install ffmpeg

# Run the app
python app.py
```

Visit http://localhost:5000

## Deployment on Render

1. Push to GitHub
2. Create new Web Service on Render
3. Select "Docker" as runtime
4. Deploy

## Tech Stack

- Flask (backend)
- librosa (audio analysis)
- matplotlib (visualization rendering)
- ffmpeg (video/audio encoding)
