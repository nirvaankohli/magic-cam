# Magic Cam
![alt text](<Screenshot 2025-10-13 154331.png>)
Real-time computer vision application that implements gesture recognition and facial feature detection for visual effects. It's also for siege!

## Overview

Computer vision application that uses deep learning and real-time image processing to detect hand gestures and facial landmarks. It also implements real-time overlay effects triggered by specific hand poses and facial detection.

## Core Features

- Gesture-based effect triggering using CNN classification
- Automated facial landmark detection for effect positioning
- Custom-trained gesture recognition model (20 classes)
- Web-based interface with configurable parameters

## Technical Architecture

- Frontend: Streamlit web application
- Computer Vision: OpenCV, MediaPipe
- Deep Learning: PyTorch, EfficientNet-B0 backbone
- Hand Detection: MediaPipe Hands with custom background subtraction
- Face Detection: MediaPipe Face Mesh
- Image Processing: PIL, NumPy arrays

## Installation

1. Clone repo:
   ```
   git clone https://github.com/nirvaankohli/magic-cam.git
   cd magic-cam
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Execute application:
   ```=
   streamlit run app.py
   ```

4. Go to `http://localhost:8501`

## Usage

1. Grant camera access permissions
2. Configure effect parameters:
   - Enable facial mesh detection for hat overlay
   - Enable gesture recognition for effect triggering
3. Execute the fireball command by pointing up
4. Export processed frames if u want

## Model Architecture

Gesture Recognition CNN:
- Input: 64x64x3 preprocessed hand region
- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Output: 20-class softmax classification
- Training:
  - Data augmentation: mixup, cutmix, random erasing
  - Optimization: SWA, early stopping
  - Cross-entropy loss with label smoothing

## Development Roadmap

- Additional gesture-effect mappings
- State-based effect system implementation
- Wand tracking via object detection
- Multi-user interaction capabilities
- Voice command integration

## Contributing

prs are appreciated!