import cv2
import threading
from deepface import DeepFace

# Initialize video capture (no CAP_DSHOW flag on macOS)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_img = cv2.imread("reference_img.jpg")

# Ensure the reference image is loaded properly
if reference_img is None:
    print("Error: Could not read the reference image.")
    exit()

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except Exception as e:
        print(f"Exception in thread: {e}")
        face_match = False

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except Exception as e:
                print(f"Error starting thread: {e}")
        counter += 1

        if face_match:
            cv2.putText(frame, "Matched!!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Not Matched", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == ord(" "):  # Use the spacebar to break
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

