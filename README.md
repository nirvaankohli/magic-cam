# Magic Cam ✨

Turn your webcam into a magical experience! This thingy lets you cast spells & wear wizard hats using just hand gestures and computer vision.

## What's This About?

Magic Cam is a real-time computer vision app that detects your hand gestures and facial features to add magical effects to your camera feed. You can point up with your finger to cast a fireball, and automatically get a wizard hat on your head.

## Features 

- **Spell Casting**: Point up with your finger to cast a fireball spell 
- **Auto Wizard Hat**: Automatically detects your face and adds a wizard hat 
- **Real-time Processing**: All effects happen live through your webcam
- **Custom CNN Model**: Trained a gesture recognition model on 20 different hand gestures
- **Streamlit Interface**: Easy-to-use web interface with effect toggles

## Tech Stack 🛠️

- **Frontend**: Streamlit (for the web interface)
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: PyTorch, custom CNN (EfficientNet-B0 backbone)
- **Hand Detection**: MediaPipe Hands + custom background subtraction
- **Face Detection**: MediaPipe Face Mesh
- **Image Processing**: PIL, NumPy

## How It Works 

1. **Hand Gesture Recognition**: Uses MediaPipe to detect hand landmarks, then feeds processed hand regions to a custom CNN trained on 20 gesture classes
2. **Face Detection**: MediaPipe Face Mesh detects facial landmarks for hat placement
3. **Effect Rendering**: OpenCV overlays magical effects (fireballs, wizard hats) on the video feed
4. **Real-time Processing**: Everything happens live

## Setup & Installation 

1. **Clone the repo**:
   ```
   git clone https://github.com/nirvaankohli/magic-cam.git
   cd magic-cam
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the Streamlit URL (usually `http://localhost:8501`)

## Usage 📸

1. **Allow camera access** when prompted
2. **Enable effects** using the sidebar:
   - Toggle "Wizard Hat" for automatic hat detection
   - Toggle "Spells" for gesture-based fireball casting
3. **Cast spells** by pointing your finger up
4. **Behind the scenes** options let you see hand/face landmarks
5. **Download** your magical photos using the download button

## Model Training 

The gesture recognition CNN was trained on a custom dataset with:
- **20 gesture classes** (numbers 0-9, hand signs, gestures)
- **EfficientNet-B0** backbone with custom classifier
- **Data augmentation**: mixup, cutmix, random erasing
- **Training techniques**: SWA (Stochastic Weight Averaging), early stopping
- **Achieved**: High accuracy on validation set

## Future Ideas 

- More spell types (shield, ice beam, summon orb)
- Dark vs Light magic system with hat color changes
- Wand tracking for more precise spell casting
- Multiplayer spell battles
- Voice commands integration

## Contributing 🤝

Feel free to open issues or submit PRs! This project is all about having fun with computer vision and magic.

---

*Made with ❤️ and a lot of OpenCV debugging loll*