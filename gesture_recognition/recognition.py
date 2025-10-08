from pathlib import Path
from hand_recognition import recognitionv2 as hrec
import cv2
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import math

root_dir = Path(__file__).parent.parent
