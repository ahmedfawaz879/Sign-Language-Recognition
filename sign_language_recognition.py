# --- sign_language_recognition.py ---
import cv2
import mediapipe as mp
import time
import math

# ----- Setup -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ------------------- UTILITIES -------------------

def distance(p1, p2):
    return math.dist(p1, p2)

def get_finger_states(lm, hand_label):
    """
    Returns list of 5 values (thumb,index,middle,ring,pinky) as 1=extended, 0=closed.
    """

    fingers = [0,0,0,0,0]

    # Thumb logic
    thumb_tip_x = lm[4][0]
    thumb_ip_x  = lm[3][0]

    if hand_label == "Right":
        if thumb_tip_x > thumb_ip_x:
            fingers[0] = 1
    else:
        if thumb_tip_x < thumb_ip_x:
            fingers[0] = 1

    # Other fingers: tip_y < pip_y → extended
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    for i in range(4):
        tip_y = lm[tip_ids[i]][1]
        pip_y = lm[pip_ids[i]][1]
        if tip_y < pip_y:
            fingers[i+1] = 1

    return fingers


# ------------------- SINGLE HAND GESTURES -------------------

def recognize_single_hand_gesture(lm, fingers, hand_label):
    thumb, idx, mid, ring, pinky = fingers

    # LIKE: thumb up only
    if thumb == 1 and idx == 0 and mid == 0 and ring == 0 and pinky == 0:
        if lm[4][1] < lm[0][1]:
            return "LIKE"

    # DISLIKE: thumb down only
    if thumb == 1 and idx == 0 and mid == 0 and ring == 0 and pinky == 0:
        if lm[4][1] > lm[0][1]:
            return "DISLIKE"

    # PUNCH: all closed
    if fingers == [0,0,0,0,0]:
        return "PUNCH"

    # VICTORY: index + pinky extended
    if idx == 1 and pinky == 1 and mid == 0 and ring == 0:
        return "VICTORY"

    # PEACE: index + middle extended
    if idx == 1 and mid == 1 and ring == 0 and pinky == 0:
        return "PEACE"

    # PERFECT: thumb touching index
    if thumb == 1 and idx == 1:
        if distance(lm[4], lm[8]) < 40:
            return "PERFECT"

    # ALLAH AKBAR: only index up
    if idx == 1 and mid == 0 and ring == 0 and pinky == 0:
        return "ALLAH_AKBAR"

    # YOU: index pointing forward (far from MCP)
    if idx == 1 and mid == 0 and ring == 0 and pinky == 0:
        if distance(lm[8], lm[5]) > 50:
            return "YOU"

    return None


# ------------------- TWO HAND GESTURES -------------------

def recognize_two_hand_gesture(hand_data):
    """
    Detects 2-hand gestures like LOVE.
    hand_data: list of {'lm': landmark list, 'fingers': list, 'label': str}
    """

    if len(hand_data) != 2:
        return None

    h1, h2 = hand_data

    # LOVE: both index + thumb tips close to each other
    idx_dist = distance(h1["lm"][8], h2["lm"][8])
    thumb_dist = distance(h1["lm"][4], h2["lm"][4])

    if idx_dist < 80 and thumb_dist < 80:
        return "LOVE"

    return None


# ------------------- MAIN LOOP -------------------

def main():
    cap = cv2.VideoCapture(0)
    p_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detected = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                hand_label = results.multi_handedness[idx].classification[0].label

                # Convert landmarks to pixel coords
                lm = []
                for lm_point in hand_landmarks.landmark:
                    lm.append((int(lm_point.x * w), int(lm_point.y * h)))

                fingers = get_finger_states(lm, hand_label)
                gesture = recognize_single_hand_gesture(lm, fingers, hand_label)

                detected.append({
                    "lm": lm,
                    "fingers": fingers,
                    "gesture": gesture,
                    "label": hand_label
                })

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # TWO HAND gesture
        multi_gesture = recognize_two_hand_gesture(detected)

        # Display gestures
        for hand in detected:
            if hand["gesture"]:
                x, y = hand["lm"][0]
                cv2.putText(frame, hand["gesture"], (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        if multi_gesture:
            cv2.putText(frame, multi_gesture, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 3)

        # Show FPS
        c_time = time.time()
        fps = int(1/(c_time - p_time))
        p_time = c_time
        cv2.putText(frame, f"FPS: {fps}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
