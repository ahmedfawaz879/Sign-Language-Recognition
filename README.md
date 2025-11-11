# Sign Language Recognition using MediaPipe Hands

This project uses **MediaPipe Hands** and **OpenCV** to perform **real‑time sign language gesture recognition** using a webcam. The system detects a set of predefined hand gestures and displays their labels directly on the camera feed.

## Features

* Real‑time gesture recognition
* Supports both single‑hand and two‑hand gestures
* Uses MediaPipe's high‑quality hand‑tracking landmarks
* Clean and extensible gesture‑logic architecture

## Recognized Gestures

### **Single‑hand Gestures**

* LIKE (thumb up)
* DISLIKE (thumb down)
* PUNCH (closed fist)
* VICTORY (index + pinky extended)
* PEACE (index + middle extended)
* PERFECT (thumb touches index)
* ALLAH_AKBAR (index finger only up)
* YOU (index extended forward)

### **Two‑hand Gesture**

* LOVE (index and thumb tips of both hands forming heart shape)

## Requirements

Install dependencies:

```bash
pip install opencv-python mediapipe
```

## How to Run

1. Connect a webcam.
2. Run the Python script:

```bash
python sign_language_recognition.py
```

3. A webcam window will open.
4. Perform gestures in front of the camera to see them recognized.
5. Press **q** to exit.

## Project Structure

```
/ (root)
  ├── sign_language_recognition.py   # Main gesture detection script
  ├── README.md                      # Project documentation
```

## How Gesture Detection Works

* MediaPipe provides **21 landmarks** per hand.
* The program extracts key distances and directions to detect gesture patterns.
* Single‑hand gestures use finger states + geometry.
* Two‑hand gestures use distances between hands.

## Customizing Gestures

All gesture logic is inside:

```python
recognize_single_hand_gesture()
recognize_two_hand_gesture()
```

You can add more gestures by analyzing landmark positions.

## Troubleshooting

* If gestures seem inverted, adjust webcam orientation.
* Low‑light conditions may reduce accuracy.
* Increase detection/tracking confidence if recognition is unstable.

## License

This project is free for personal and educational use.

## Contributions

Feel free to open pull requests with:

* New gesture implementations
* Code improvements
* Documentation updates

Happy hacking!
