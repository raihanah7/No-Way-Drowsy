import cv2
import urllib.request

SERVER_URL = "http://127.0.0.1:8765/predict"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    raise SystemExit

print("Camera opened.")
print("Press SPACE to capture one frame.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame")
        break

    cv2.imshow("YOLO Server Test - Press SPACE", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE
        print("Sending frame to YOLO server...")

        success, encoded = cv2.imencode(".jpg", frame)

        if not success:
            print("ERROR: Could not encode frame")
            break

        request = urllib.request.Request(
            SERVER_URL,
            data=encoded.tobytes(),
            headers={
                "Content-Type": "image/jpeg"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                result = response.read().decode("utf-8")

            print("\nYOLO RESPONSE:")
            print(result)

        except Exception as e:
            print("\nERROR connecting to YOLO server:")
            print(e)

        break

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()