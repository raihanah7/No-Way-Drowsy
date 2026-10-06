from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import cv2
import numpy as np
import threading
import time
import winsound
from ultralytics import YOLO


HOST = "127.0.0.1"
PORT = 8765
MODEL_PATH = "best.onnx"
CONF_THRESHOLD = 0.5


# Project thresholds
CLOSED_FRAMES_THRESHOLD = 10
PERCLOS_WINDOW = 90
PERCLOS_THRESHOLD = 0.70
PERCLOS_CRITICAL = 1.00


# Alert sound state
ALERT_SOUND_PLAYING = False
ALERT_SOUND_ACTIVE = False
ALERT_SOUND_CRITICAL = False


def alert_sound_loop():
    global ALERT_SOUND_PLAYING

    while ALERT_SOUND_ACTIVE:

        if ALERT_SOUND_CRITICAL:
            winsound.Beep(1200, 250)
            time.sleep(0.08)

            winsound.Beep(1200, 250)
            time.sleep(0.08)

        else:
            winsound.Beep(880, 400)
            time.sleep(0.15)

    ALERT_SOUND_PLAYING = False


def update_alert_sound(active, critical=False):
    global ALERT_SOUND_ACTIVE
    global ALERT_SOUND_CRITICAL
    global ALERT_SOUND_PLAYING

    ALERT_SOUND_ACTIVE = active
    ALERT_SOUND_CRITICAL = critical

    if active and not ALERT_SOUND_PLAYING:

        ALERT_SOUND_PLAYING = True

        threading.Thread(
            target=alert_sound_loop,
            daemon=True
        ).start()


model = YOLO(MODEL_PATH)


# Latest valid YOLO11 result
LATEST_RESULT = {
    "ok": True,
    "eye_state": "NO DETECTION",
    "confidence": 0.0,
    "detections": [],
    "closed_streak": 0,
    "perclos": 0.0,
    "alert_status": "READY",
    "alert_active": False,
}


# Current closed-eye frame streak
CLOSED_STREAK = 0


# Store eye state for the latest inference frames
PERCLOS_HISTORY = []


class Handler(BaseHTTPRequestHandler):

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.send_header(
            "Content-Length",
            str(len(data))
        )

        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send_json({"ok": True})

    def do_GET(self):
        if self.path == "/latest":
            self._send_json(LATEST_RESULT)
            return

        self._send_json(
            {"error": "Not found"},
            404
        )

    def do_POST(self):
        global LATEST_RESULT
        global CLOSED_STREAK
        global PERCLOS_HISTORY

        if self.path != "/predict":
            self._send_json(
                {"error": "Not found"},
                404
            )
            return

        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            body = self.rfile.read(length)

            frame = cv2.imdecode(
                np.frombuffer(
                    body,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                self._send_json(
                    {"error": "Invalid image"},
                    400
                )
                return

            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=320,
                verbose=False,
                device="cpu",
            )

            result = results[0]

            detections = []

            for box in result.boxes:
                cls_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                name = result.names[cls_id]

                xyxy = box.xyxy[0].tolist()

                detections.append({
                    "class": name,
                    "confidence": round(
                        confidence,
                        3
                    ),
                    "box": [
                        round(v, 1)
                        for v in xyxy
                    ],
                })

            # Get the highest-confidence detection
            best = max(
                detections,
                key=lambda x: x["confidence"],
                default=None,
            )

            # Update only when YOLO11 has a valid detection
            if best:
                eye_state = best["class"]

                # Update closed-eye streak
                if eye_state.lower() == "closed":
                    CLOSED_STREAK += 1

                elif eye_state.lower() == "open":
                    CLOSED_STREAK = 0

                # Update PERCLOS history
                if eye_state.lower() == "closed":
                    PERCLOS_HISTORY.append(1)

                elif eye_state.lower() == "open":
                    PERCLOS_HISTORY.append(0)

                # Keep only the latest 90 valid eye-state frames
                if len(PERCLOS_HISTORY) > PERCLOS_WINDOW:
                    PERCLOS_HISTORY = PERCLOS_HISTORY[
                        -PERCLOS_WINDOW:
                    ]

                # Calculate PERCLOS
                if PERCLOS_HISTORY:
                    perclos = (
                        sum(PERCLOS_HISTORY)
                        / len(PERCLOS_HISTORY)
                    )
                else:
                    perclos = 0.0

                # Determine alert status
                if perclos >= PERCLOS_CRITICAL:
                    alert_status = "CRITICAL"
                    alert_active = True

                    update_alert_sound(
                        active=True,
                        critical=True
                    )

                elif (
                    CLOSED_STREAK >= CLOSED_FRAMES_THRESHOLD
                    or perclos >= PERCLOS_THRESHOLD
                ):
                    alert_status = "ACTIVE"
                    alert_active = True

                    update_alert_sound(
                        active=True,
                        critical=False
                    )

                else:
                    alert_status = "READY"
                    alert_active = False

                    update_alert_sound(
                        active=False,
                        critical=False
                    )

                LATEST_RESULT = {
                    "ok": True,
                    "eye_state": eye_state,
                    "confidence": best["confidence"],
                    "detections": detections,
                    "closed_streak": CLOSED_STREAK,
                    "perclos": round(
                        perclos,
                        3
                    ),
                    "alert_status": alert_status,
                    "alert_active": alert_active,
                }

            self._send_json(
                LATEST_RESULT
            )

        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                },
                500
            )

    def log_message(
        self,
        format,
        *args
    ):
        # Keep the terminal clean.
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler
    )

    print(
        f"YOLO11 server running at "
        f"http://{HOST}:{PORT}"
    )

    print(
        "POST endpoint: /predict"
    )

    print(
        "GET endpoint: /latest"
    )

    print(
        "Press Ctrl+C to stop."
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()