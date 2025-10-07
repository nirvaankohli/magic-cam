# Project Concept Draft

#### Core Idea:

A computer vision based interactive system where a user casts using _hand gestures, body motion, etc._ Each gesture or event triggers unique visual effects. Ex. fireballs, shields or levitation. This will be displayed in a streamlit app(deployed [Here](https://share.streamlit.io/)) where the webcam video stream will be sent to the backend(python) where it will be processed and sent back to display on the website.

## Technicals

1. Input Capture
   - Web Camera or external camera(connected to user's device)
   - This data is then sent through streamlit and into the backend processing thingy
2. Gesture Recognition
   - Use computer vision & ML to detect gestures, facial movements, etc.
   - Options of stuff we can consider
     - Thinking of using opencv to map out points on the user's hand
     - Then map these points and use a ML model to recognize these landmarks.
   - For head thing use the same thing as the hand(should be easier)
7777
3. Effects
   - Us an animation layer via cv2 overlays
   - Each recognized gesture triggers things like sparkles, images, sound effects, etc.

## Spell mapping(still in progress)

| Gesture               | Spell      |
| --------------------- | ---------- |
| Palm Foward           | Fireball   |
| Both hands up         | Shield     |
| Circle-ish movement   | Summon Orb |
| Finger gun(rlly hard) | Ice Beam   |


# Architecture

### Ideas to make it better

- Add like bars for each type of spell
  - 2 types of spells: Dark & Light.
    - If you have used more dark spells your hat turns black
- Wand tracking hand
  - Cool, ig?
