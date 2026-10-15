# Parking System

A modular parking lot vehicle counting framework built for official PyTorch RT-DETR, ByteTrack, and OpenCV.

## Purpose

This project provides a modular baseline for counting vehicles entering and leaving three parking zones:

- A: Front
- B: Left
- C: Right

It is designed so that video input, detection, tracking, counting, and visualization can be replaced independently.

## Project Structure

- `main.py`: application entry point
- `config.yaml`: all runtime parameters and default zone settings
- `detector/`: RT-DETR detector wrapper
- `tracker/`: ByteTrack wrapper
- `counting/`: zone counting logic
- `video/`: MP4 and future RTSP input abstraction
- `visualization/`: OpenCV drawing utilities
- `utils/`: geometry helpers
- `assets/test.mp4`: default test video input
- `weights/`: model weights directory
- `output/`: annotated output video directory

## Environment

Recommended:

- Python 3.10
- Conda environment named `parking-rtdetr`
- PyTorch
- OpenCV
- NumPy
- PyYAML

CPU-only setup for this project:

```bash
conda create -n parking-rtdetr python=3.10 -y
conda activate parking-rtdetr
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt
```

If the official RT-DETR package needs to be installed separately, use the GitHub source listed in `requirements.txt`.

## Installation

```bash
pip install -r requirements.txt
```

For CPU-only execution, install the CPU wheels for PyTorch first, then install the remaining requirements.

## RT-DETR Weights

Place the official RT-DETR weights in:

- `weights/model.pth`

Update `config.yaml` if your local path differs.

## Test Video

Place the test video in:

- `assets/test.mp4`

## Configuration

Edit `config.yaml` to change:

- parking capacities
- virtual line coordinates
- detection confidence threshold
- tracking thresholds
- video source

## Run

```bash
python main.py
```

Press `q` to exit.

## Switching to RTSP

Change `video.type` and `video.source` in `config.yaml` from file input to your RTSP stream. The detector, tracker, and counter modules do not need to be rewritten.

## ENTER / EXIT

The zone counter is designed to detect line crossing events and update occupied counts when a vehicle enters or exits a zone. The current scaffold includes the module boundaries and state storage needed for that logic.

## Known Limits

- RT-DETR inference is not wired yet.
- ByteTrack association is not wired yet.
- Zone crossing logic is scaffolded and will need the final event rules.
- The default test video is not included.
- The default weights file is not included.
