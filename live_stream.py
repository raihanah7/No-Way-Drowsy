"""
=============================================================
  DROWSINESS DETECTION SYSTEM
  Using YOLO11 fine-tuned model (best.pt) + Webcam
  Classes: 'open' (eyes open) | 'closed' (eyes closed)
=============================================================
  Requirements:
    pip install ultralytics opencv-python numpy pygame
=============================================================
"""

import cv2
import numpy as np
import time
import pygame
import os
from ultralytics import YOLO
from collections import deque

# ──────────────────────────────────────────────
#  CONFIGURATION 
# ──────────────────────────────────────────────

MODEL_PATH       = "best.pt"
CONF_THRESHOLD   = 0.5              # Minimum confidence to count a detection
CAMERA_INDEX     = 0                # 0 = default webcam, 1 = external webcam

# Drowsiness logic
EYE_CLOSED_CLASS = "closed"
EYE_OPEN_CLASS   = "open"

CLOSED_FRAMES_THRESHOLD = 20        # Frames of closed eyes before alert triggers
PERCLOS_WINDOW   = 90               # Sliding window (frames) for PERCLOS metric
PERCLOS_THRESHOLD = 0.7             # Alert if >70% of window frames are "closed"
PERCLOS_CRITICAL  = 1.0             # FULL ALARM threshold (PERCLOS = 100%)

# Alert sound
ALERT_SOUND_PATH = None

# Display
WINDOW_TITLE     = "Drowsiness Detection | Press Q to quit"
FONT             = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────
#  COLORS (BGR)
# ──────────────────────────────────────────────
COLOR_OPEN    = (50, 220, 100)
COLOR_CLOSED  = (50, 100, 255)
COLOR_ALERT   = (0, 0, 255)
COLOR_CRITICAL= (0, 0, 200)
COLOR_SAFE    = (50, 220, 100)
COLOR_TEXT    = (255, 255, 255)
COLOR_BG      = (20, 20, 30)


# ──────────────────────────────────────────────
#  BUILT-IN BEEP GENERATOR (no wav or mp3 file needed)
# ──────────────────────────────────────────────
SAMPLE_RATE = 44100

def _generate_beep(frequency=880, duration=0.4, volume=0.9):
    """Generate a sharp sine-wave beep as a pygame Sound object."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
    # Make stereo
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

def _generate_alarm(frequency=1100, duration=0.25, repeats=3, gap=0.08, volume=1.0):
    """Generate an urgent repeating alarm burst (for PERCLOS = 100%)."""
    t_on  = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    t_gap = np.zeros(int(SAMPLE_RATE * gap), dtype=np.int16)
    beep  = (np.sin(2 * np.pi * frequency * t_on) * volume * 32767).astype(np.int16)
    silence = t_gap
    burst = np.concatenate([np.concatenate([beep, silence]) for _ in range(repeats)])
    stereo = np.column_stack([burst, burst])
    return pygame.sndarray.make_sound(stereo)


# ──────────────────────────────────────────────
#  ALERT SOUND SETUP
# ──────────────────────────────────────────────
def init_sound():
    """Initialize pygame mixer and return (normal_sound, critical_alarm)."""
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    # Normal alert beep
    if ALERT_SOUND_PATH and os.path.exists(ALERT_SOUND_PATH):
        normal = pygame.mixer.Sound(ALERT_SOUND_PATH)
    else:
        normal = _generate_beep(frequency=880, duration=0.4, volume=0.8)

    # Critical 100% alarm — louder, faster, more urgent
    critical = _generate_alarm(frequency=1200, duration=0.2, repeats=4, gap=0.06, volume=1.0)

    return normal, critical


def play_alert(sounds, alert_active, critical_active):
    """
    Play appropriate sound based on alert level.
    critical_active = PERCLOS reached 100%.
    alert_active    = regular drowsiness threshold crossed.
    """
    normal_sound, critical_sound = sounds

    if critical_active:
        # Stop normal sound, play critical alarm if not already playing
        if not pygame.mixer.get_busy():
            critical_sound.play()
    elif alert_active:
        if not pygame.mixer.get_busy():
            normal_sound.play()
    else:
        pygame.mixer.stop()


# ──────────────────────────────────────────────
#  DRAWING UTILITIES
# ──────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=10, thickness=2):
    """Draw a rounded rectangle (bounding box)."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_status_bar(frame, h, w, eye_state, consecutive_closed, perclos, alert, critical):
    """Draw the HUD at the bottom of the frame."""
    bar_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Eye state
    state_color = COLOR_CLOSED if eye_state == EYE_CLOSED_CLASS else COLOR_OPEN
    state_label = f"Eyes: {eye_state.upper()}" if eye_state else "Eyes: ---"
    cv2.putText(frame, state_label, (15, h - 58), FONT, 0.7, state_color, 2)

    # Consecutive closed frames
    cv2.putText(frame, f"Closed streak: {consecutive_closed}/{CLOSED_FRAMES_THRESHOLD}",
                (15, h - 28), FONT, 0.55, COLOR_TEXT, 1)

    # PERCLOS bar
    bar_x, bar_y, bar_w_total, bar_h_inner = w // 2, h - 68, 240, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_total, bar_y + bar_h_inner), (60, 60, 80), -1)
    filled = int(perclos * bar_w_total)
    if critical:
        pbar_color = COLOR_CRITICAL
    elif perclos >= PERCLOS_THRESHOLD:
        pbar_color = COLOR_ALERT
    else:
        pbar_color = COLOR_OPEN
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h_inner), pbar_color, -1)
    cv2.putText(frame, f"PERCLOS: {perclos*100:.0f}%", (bar_x, bar_y - 6), FONT, 0.5, COLOR_TEXT, 1)

    # Alert status
    if critical:
        alert_text  = "!! CRITICAL — 100% DROWSY !!"
        alert_color = COLOR_CRITICAL
    elif alert:
        alert_text  = "WARNING: DROWSINESS DETECTED"
        alert_color = COLOR_ALERT
    else:
        alert_text  = "OK  Normal"
        alert_color = COLOR_SAFE
    cv2.putText(frame, alert_text, (w - 340, h - 45), FONT, 0.65, alert_color, 2)

def draw_alert_overlay(frame, h, w, critical=False):
    """Flashing red border (deeper red + full screen tint when critical)."""
    color = COLOR_CRITICAL if critical else COLOR_ALERT
    overlay = frame.copy()
    if critical:
        # Subtle full red tint
        cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        overlay = frame.copy()
    for thickness in [40, 25, 12]:
        cv2.rectangle(overlay, (0, 0), (w, h), color, thickness)
    alpha = 0.55 if critical else 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if critical:
        msg = "!! WAKE UP !!"
        (tw, th), _ = cv2.getTextSize(msg, FONT, 2.2, 4)
        cx, cy = (w - tw) // 2, h // 2
        cv2.putText(frame, msg, (cx + 3, cy + 3), FONT, 2.2, (0, 0, 0), 6)
        cv2.putText(frame, msg, (cx, cy), FONT, 2.2, (0, 0, 220), 4)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), FONT, 0.55, (180, 180, 180), 1)


# ──────────────────────────────────────────────
#  MAIN DETECTION LOOP
# ──────────────────────────────────────────────
def main():
    print("[INFO] Loading YOLO11 model...")
    model = YOLO(MODEL_PATH)
    class_names = model.names  # {0: 'closed', 1: 'open'}
    print(f"[INFO] Model classes: {class_names}")

    label_to_id = {v: k for k, v in class_names.items()}
    print(f"[INFO] Label map: {label_to_id}")

    sounds = init_sound()   # returns (normal_beep, critical_alarm)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check CAMERA_INDEX.")
        return

    print("[INFO] Starting detection. Press Q to quit.\n")

    # State tracking
    consecutive_closed  = 0
    alert_active        = False
    critical_active     = False
    perclos_window      = deque(maxlen=PERCLOS_WINDOW)
    fps_counter         = 0
    fps_display         = 0.0
    fps_timer           = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        h, w = frame.shape[:2]

        # ── Run YOLO inference ──
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            verbose=False,
            stream=False
        )

        # ── Parse detections ──
        eye_state    = None
        best_conf    = 0.0
        frame_closed = False

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id    = int(box.cls[0])
                conf      = float(box.conf[0])
                label     = class_names.get(cls_id, "unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = COLOR_CLOSED if label == EYE_CLOSED_CLASS else COLOR_OPEN
                draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=8, thickness=2)

                tag = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, FONT, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, tag, (x1 + 3, y1 - 4), FONT, 0.55, (0, 0, 0), 1)

                if conf > best_conf:
                    best_conf = conf
                    eye_state = label

                if label == EYE_CLOSED_CLASS:
                    frame_closed = True

        # ── Update state counters ──
        perclos_window.append(1 if frame_closed else 0)
        perclos = sum(perclos_window) / len(perclos_window) if perclos_window else 0.0

        if frame_closed:
            consecutive_closed += 1
        else:
            consecutive_closed = 0

        # ── Determine alert levels ──
        # Level 1 — warning: streak OR PERCLOS ≥ 70%
        alert_active = (
            consecutive_closed >= CLOSED_FRAMES_THRESHOLD or
            perclos >= PERCLOS_THRESHOLD
        )
        # Level 2 — critical: PERCLOS = 100% (all frames in window are closed)
        critical_active = (perclos >= PERCLOS_CRITICAL)

        # ── Audio alarm ──
        play_alert(sounds, alert_active, critical_active)

        # ── Draw overlays ──
        if critical_active or alert_active:
            draw_alert_overlay(frame, h, w, critical=critical_active)

        draw_status_bar(frame, h, w, eye_state, consecutive_closed,
                        perclos, alert_active, critical_active)

        # ── FPS counter ──
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer   = time.time()
        draw_fps(frame, fps_display)

        # ── Console log ──
        if critical_active:
            status = "CRITICAL"
        elif alert_active:
            status = "ALERT"
        else:
            status = "OK"

        print(f"\r[{status}] Eye: {str(eye_state):<8} | "
              f"Streak: {consecutive_closed:>3}/{CLOSED_FRAMES_THRESHOLD} | "
              f"PERCLOS: {perclos*100:>5.1f}% | FPS: {fps_display:.1f}  ",
              end="", flush=True)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    print("\n[INFO] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()



if __name__ == "__main__":
    main()
