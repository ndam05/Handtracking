import cv2
import time
from pathlib import Path
import mediapipe as mp
import json

BASE = Path(__file__).parent
MODEL_PATH = str(BASE / "models" / "hand_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

current_result = None
frame_count = 0

def callback(result, output_image, timestamp_ms):
    global current_result
    current_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=callback,
    num_hands=1
)

def get_hand_data():
    """Return hand data in the required format"""
    if current_result and current_result.hand_landmarks:
        hand = current_result.hand_landmarks[0]
    
    
        middle_finger_Bottom_x = hand[9].x
        middle_finger_Bottom_y = hand[9].y
        
        # Hand is detected
        hand_detected = True
        
        return {
            "hand_detected": hand_detected,
            "middle_finger_Bottom": {
                "x": round(middle_finger_Bottom_x, 4),
                "y": round(middle_finger_Bottom_y, 4)
            }
        }
    else:
        return {
            "hand_detected": False,
            "middle_finger_Bottom": {
                "x": 0,
                "y": 0
            }
        }

def main():
    global frame_count
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Geen webcam gevonden")
        return

    print("Hand tracking gestart")
    print("=" * 50)

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, int(time.time() * 1000))
            data = get_hand_data()
            print(json.dumps(data))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nHand tracking gestopt")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()