import streamlit as st

import streamlit.components.v1 as components







import textwrap







import base64



import urllib.request

import json



from pathlib import Path

import cv2

import numpy as np

from ultralytics import YOLO

import threading



import threading



import time







import av



import cv2



from ultralytics import YOLO



from streamlit_webrtc import webrtc_streamer, WebRtcMode























# =========================================================*







# ASSETS*







# =========================================================*















BASE_DIR = Path(__file__).resolve().parent







LOGO_PATH = BASE_DIR / "logo.png"
ROAD_PATH = BASE_DIR / "logo2.png"
ROAD_BG_PATH = BASE_DIR / "road_background.png"
# =========================================================*

# YOLO11 + LOCAL CAMERA*

# =========================================================*



MODEL_PATH = BASE_DIR / "best.pt"



_yolo_model = None

_yolo_model_lock = threading.Lock()



if MODEL_PATH.exists():

    try:

        _yolo_model = YOLO(str(MODEL_PATH))

    except Exception:

        _yolo_model = None





def process_camera_frame(image):

    """Run lightweight YOLO11 inference on a BGR frame."""

    if _yolo_model is None:

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    try:

        with _yolo_model_lock:

            results = _yolo_model.predict(

                source=image,

                conf=0.5,

                imgsz=320,

                verbose=False,

                device="cpu",

                max_det=20,

            )

        if results:

            image = results[0].plot(conf=True, line_width=2)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception:

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





def open_camera():

    if st.session_state.get("opencv_camera") is None:

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():

            cap.release()

            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        cap.set(cv2.CAP_PROP_FPS, 15)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        st.session_state.opencv_camera = cap

    return st.session_state.opencv_camera





def close_camera():

    cap = st.session_state.get("opencv_camera")

    if cap is not None:

        cap.release()

    st.session_state.opencv_camera = None





for _key, _value in {

    "opencv_camera": None,

    "camera_frame_count": 0,

    "last_yolo_frame": None,

}.items():

    if _key not in st.session_state:

        st.session_state[_key] = _value





def live_camera():

    """Use the browser's native camera preview for smooth, zero-rerun video."""

    camera_enabled = str(st.session_state.camera_running).lower()

    html = """

        <style>
            html, body {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background: transparent;
            }

            .camera {
                width: 100%;
                height: 473px;
                margin: 0;
                border-radius: 18px;
                overflow: hidden;
                background: linear-gradient(
                    135deg,
                    #dfe5e7 0%,
                    #bfc9cf 45%,
                    #8e9ca5 100%
                );
                position: relative;
            }

            video {
                width: 100%;
                height: 100%;
                display: block;
                object-fit: cover;
                transform: scaleX(-1);
                background: #8e9ca5;
            }

            .message {
                position: absolute;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-family: Arial, sans-serif;
                font-size: 17px;
                font-weight: 600;
                text-align: center;
                pointer-events: none;
            }
        </style>

        <div class="camera">
            <video id="video" autoplay muted playsinline></video>
            <div id="message" class="message">Starting camera...</div>
        </div>

        <script>
            const video = document.getElementById("video");
            const message = document.getElementById("message");

            let stream = null;
            let yoloTimer = null;
            let yoloBusy = false;

            const YOLO_SERVER = "http://127.0.0.1:8765/predict";

            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");

            async function analyzeFrame() {
                if (!stream || video.readyState < 2 || yoloBusy) {
                    return;
                }

                yoloBusy = true;

                try {
                    canvas.width = 320;
                    canvas.height = 320;

                    ctx.drawImage(video, 0, 0, 320, 320);

                    const blob = await new Promise(resolve => {
                        canvas.toBlob(resolve, "image/jpeg", 0.6);
                    });

                    if (!blob) {
                        return;
                    }

                    const response = await fetch(YOLO_SERVER, {
                        method: "POST",
                        headers: {
                            "Content-Type": "image/jpeg"
                        },
                        body: blob
                    });

                    if (!response.ok) {
                        throw new Error(
                            "YOLO server returned HTTP " + response.status
                        );
                    }

                    const data = await response.json();

                    localStorage.setItem(
                        "nwd_eye_state",
                        data.eye_state || "NO DETECTION"
                    );

                    localStorage.setItem(
                        "nwd_eye_confidence",
                        data.confidence || 0
                    );

                    console.log(
                        "[No Way Drowsy] YOLO11:",
                        data.eye_state,
                        data.confidence
                    );

                } catch (error) {
                    console.error(
                        "[No Way Drowsy] YOLO connection error:",
                        error
                    );
                } finally {
                    yoloBusy = false;
                }
            }

            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            width: { ideal: 640 },
                            height: { ideal: 360 },
                            frameRate: { ideal: 20, max: 30 }
                        },
                        audio: false
                    });

                    video.srcObject = stream;
                    await video.play();

                    message.style.display = "none";

                    yoloTimer = setInterval(analyzeFrame, 500);

                } catch (error) {
                    message.textContent =
                        "Camera access was blocked. Please allow camera access.";
                    console.error(error);
                }
            }

            function stopCamera() {
                if (yoloTimer) {
                    clearInterval(yoloTimer);
                    yoloTimer = null;
                }

                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }

                video.srcObject = null;
                message.style.display = "flex";
                message.textContent = "Monitoring stopped.";
            }

            const CAMERA_ENABLED = CAMERA_STATE;

            if (CAMERA_ENABLED) {
                startCamera();
            } else {
                stopCamera();
            }

            window.addEventListener("beforeunload", () => {
                if (yoloTimer) {
                    clearInterval(yoloTimer);
                }

                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
            });
        </script>
    """

    html = html.replace("CAMERA_STATE", camera_enabled)

    components.html(
        html,
        height=493,
        scrolling=False,
    )





# =========================================================*



# YOLO11*



# =========================================================*







MODEL_PATH = BASE_DIR / "best.pt"







_yolo_model = None



_yolo_model_lock = threading.Lock()







if MODEL_PATH.exists():



    try:



        _yolo_model = YOLO(str(MODEL_PATH))



    except Exception:



        _yolo_model = None











def process_yolo_frame(frame):



    """Run YOLO11 on a camera frame and return an annotated frame."""



    image = frame.to_ndarray(format="bgr24")



    image = cv2.flip(image, 1)







    if _yolo_model is None:



        cv2.putText(



            image,



            "YOLO11 model not found",



            (20, 40),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.8,



            (255, 255, 255),



            2,



            cv2.LINE_AA,



        )



        return av.VideoFrame.from_ndarray(image, format="bgr24")







    try:



        with _yolo_model_lock:



            results = _yolo_model.predict(



                source=image,



                conf=0.5,



                imgsz=640,



                verbose=False,



                device="cpu",



            )







        if results:



            result = results[0]



            image = result.plot()







        return av.VideoFrame.from_ndarray(image, format="bgr24")







    except Exception as exc:



        cv2.putText(



            image,



            f"YOLO11 error: {str(exc)[:45]}",



            (20, 40),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.6,



            (255, 255, 255),



            2,



            cv2.LINE_AA,



        )



        return av.VideoFrame.from_ndarray(image, format="bgr24")























# =========================================================*







# PAGE CONFIG*







# =========================================================*















st.set_page_config(







    page_title="No Way Drowsy",







    page_icon=str(LOGO_PATH),







    layout="wide",







    initial_sidebar_state="collapsed",







)

ROAD_BG_PATH = BASE_DIR / "road_background.png"


def get_road_bg_data_uri():
    if ROAD_BG_PATH.exists():
        encoded = base64.b64encode(ROAD_BG_PATH.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    return ""


ROAD_BG_DATA_URI = get_road_bg_data_uri()

def get_logo_data_uri():







    if LOGO_PATH.exists():







        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")







        return f"data:image/png;base64,{encoded}"







    return ""























def get_road_data_uri():







    if ROAD_PATH.exists():







        encoded = base64.b64encode(ROAD_PATH.read_bytes()).decode("utf-8")







        return f"data:image/png;base64,{encoded}"







    return ""
def get_road_bg_data_uri():
    if ROAD_BG_PATH.exists():
        encoded = base64.b64encode(
            ROAD_BG_PATH.read_bytes()
        ).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    return ""


ROAD_BG_DATA_URI = get_road_bg_data_uri()























LOGO_DATA_URI = get_logo_data_uri()







ROAD_DATA_URI = get_road_data_uri()























# =========================================================*







# HTML HELPER*







# =========================================================*







# st.html is used instead of st.markdown for custom HTML.*







# This prevents Streamlit from displaying HTML as code text.*















def render_html(content):







    st.html(textwrap.dedent(content).strip())























# =========================================================*







# CUSTOM CSS*







# =========================================================*















st.html("""







<style>















    /* =====================================================







       MAIN PAGE







       ===================================================== */















    .stApp {







        background: #f7f5f1;







    }















    .main .block-container,







    .stMainBlockContainer {







        max-width: 1500px;







        padding-top: 6px !important;







        padding-bottom: 18px;







        padding-left: 28px;







        padding-right: 28px;







    }















    section.main> div {







        padding-top: 0 !important;







    }















    #MainMenu,







    footer,







    header {







        visibility: hidden;







    }























    /* =====================================================







       HEADER







       ===================================================== */















    .nwd-header {







        display: flex;







        align-items: center;







        justify-content: space-between;







        margin-bottom: 14px;







        padding: 0 34px;







        min-height: 122px;







    }















    .nwd-brand {







        display: flex;







        align-items: center;







        gap: 24px;







        flex: 1 1 auto;







        min-width: 0;







    }















    .nwd-brand-text {







        flex: 1 1 auto;







        min-width: 0;







        overflow: visible;







        padding-right: 10px;







    }















    .nwd-logo {







        width: 132px;







        height: 132px;







        object-fit: contain;







        display: block;







        flex: 0 0 132px;







    }















    .nwd-title {







        font-size: 46px;







        font-weight: 800;







        color: #27313b;







        line-height: 1.05;







        letter-spacing: -0.4px;







        margin-bottom: 8px;







        white-space: nowrap;







        display: block;







        width: max-content;







        transform: scaleX(1.22);







        transform-origin: left center;







    }















    .nwd-subtitle {







        font-size: 21px;







        color: #7b838b;







        font-weight: 450;







        line-height: 1.2;







        letter-spacing: 0.15px;







        white-space: nowrap;







        display: block;







        width: max-content;







        transform: scaleX(1.22);







        transform-origin: left center;







    }















    .nwd-header-right {







        position: relative;







        width: 610px;







        height: 132px;







        display: flex;







        align-items: center;







        justify-content: flex-end;







        overflow: visible;







    }















    .nwd-road {







        position: absolute;







        right: -285px;







        top: -10px;







        width: 880px;







        height: 154px;







        object-fit: fill;







        object-position: center;







        opacity: 0.92;







        z-index: 0;















        /* Let the road blend naturally into the page on the left,







           while extending beyond the right side so no hard vertical edge appears. */







        -webkit-mask-image: linear-gradient(







            90deg,







            transparent 0%,







            rgba(0,0,0,0.12) 10%,







            rgba(0,0,0,0.48) 22%,







            black 38%,







            black 76%,







            rgba(0,0,0,0.92) 82%,







            rgba(0,0,0,0.45) 90%,







            transparent 98%,







            transparent 100%







        );







        mask-image: linear-gradient(







            90deg,







            transparent 0%,







            rgba(0,0,0,0.12) 10%,







            rgba(0,0,0,0.48) 22%,







            black 38%,







            black 76%,







            rgba(0,0,0,0.92) 82%,







            rgba(0,0,0,0.45) 90%,







            transparent 98%,







            transparent 100%







        );







    }















    .nwd-slogan {







        position: relative;







        z-index: 2;







        display: flex;







        align-items: center;







        gap: 16px;







        color: #68727c;







        font-size: 16px;







        font-weight: 650;







        line-height: 1.25;







        margin-right: 360px;







        white-space: nowrap;







    }















    .nwd-slogan-line {







        width: 50px;







        height: 2px;







        background: #78a5c5;







    }























    /* =====================================================







       GENERAL CARDS







       ===================================================== */















    .nwd-card {







        background: rgba(255, 255, 255, 0.90);







        border: 1px solid #e5e5e2;







        border-radius: 20px;







        box-shadow: 0 3px 14px rgba(60, 70, 80, 0.08);







        box-sizing: border-box;







    }























    /* =====================================================







       CAMERA CARD







       ===================================================== */















    .nwd-camera-card {







        padding: 0;







        overflow: hidden;







        height: 548px;







    }















    .nwd-camera-header {







        height: 58px;







        display: flex;







        align-items: center;







        justify-content: space-between;







        padding: 0 20px;







        box-sizing: border-box;







    }















    .nwd-camera-title {







        display: flex;







        align-items: center;







        gap: 10px;







        color: #27313b;







        font-size: 17px;







        font-weight: 700;







    }















    .nwd-detecting {







        display: flex;







        align-items: center;







        gap: 8px;







        border: 1px solid #eeeeeb;







        background: #fafaf8;







        border-radius: 20px;







        padding: 8px 14px;







        color: #606970;







        font-size: 14px;







    }















    .nwd-green-dot {







        width: 10px;







        height: 10px;







        border-radius: 50%;







        background: #43ad6b;







        display: inline-block;







    }















    .nwd-camera-preview {







        height: 473px;







        margin: 0 14px 10px;







        border-radius: 18px;







        background: linear-gradient(







            135deg,







            #dfe5e7 0%,







            #bfc9cf 45%,







            #8e9ca5 100%







        );







        position: relative;







        overflow: hidden;







        display: flex;







        align-items: center;







        justify-content: center;







    }















    .nwd-camera-placeholder {







        text-align: center;







        color: white;







        opacity: 0.92;







    }















    .nwd-camera-icon {







        font-size: 55px;







        margin-bottom: 10px;







    }















    .nwd-camera-text {







        font-size: 18px;







        font-weight: 600;







    }















    .nwd-camera-subtext {







        font-size: 13px;







        opacity: 0.8;







        margin-top: 5px;







    }















    .nwd-camera-badge {







        position: absolute;







        top: 16px;







        left: 16px;







        background: #48b877;







        color: white;







        padding: 8px 13px;







        border-radius: 10px;







        font-size: 14px;







        font-weight: 600;







    }















    .nwd-fps-badge {







        position: absolute;







        bottom: 14px;







        left: 14px;







        background: rgba(30, 35, 40, 0.82);







        color: white;







        padding: 8px 12px;







        border-radius: 10px;







        font-size: 13px;







    }























    /* =====================================================







       DRIVER STATUS







       ===================================================== */















    .nwd-status-card {







        padding: 0;







        margin-bottom: 8px;







    }















    .nwd-status-header {







        padding: 11px 20px 6px;







        display: flex;







        align-items: center;







        gap: 10px;







        color: #27313b;







        font-size: 16px;







        font-weight: 700;







    }















    .nwd-status-body {







        margin: 0 14px 10px;







        border-radius: 17px;







        background: #dcefdc;







        min-height: 72px;







        display: flex;







        align-items: center;







        justify-content: center;







        gap: 14px;







    }















    .nwd-status-circle {







        width: 29px;







        height: 29px;







        border-radius: 50%;







        border: 4px solid #68bd88;







        box-shadow: 0 0 0 4px rgba(104, 189, 136, 0.22);







        background: #bfe2c9;







    }















    .nwd-status-main {







        font-size: 27px;







        font-weight: 800;







        color: #27313b;







    }















    .nwd-status-sub {







        text-align: center;







        font-size: 14px;







        color: #4d5962;







        margin-top: 2px;







    }























    /* =====================================================







       METRICS







       ===================================================== */















    .nwd-metric-card {
        background: #edf2f7;
        border: 1px solid #e5e9ed;







        border-radius: 16px;







        min-height: 96px;







        padding: 11px 15px;







        margin-bottom: 8px;







        box-sizing: border-box;







    }
   
    .nwd-bottom-road {
        width: 760px;
        height: 230px;
        margin-top: -25px;
        margin-left: -430px;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: flex-end;
    }
    .nwd-bottom-road img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center bottom;
        display: block;
    }

    .nwd-metric-label {







        color: #56616b;







        font-size: 14px;







        font-weight: 600;







        display: flex;







        align-items: center;







        gap: 8px;







    }





    









    .nwd-metric-value {







        color: #27313b;







        font-size: 27px;







        font-weight: 750;







        margin-top: 6px;







    }















    .nwd-metric-small {







        color: #6d757d;







        font-size: 12px;







    }















    .nwd-progress-bg {







        width: 100%;







        height: 8px;







        border-radius: 8px;







        background: #d9dde1;







        margin-top: 6px;







        overflow: hidden;







    }















    .nwd-progress-fill {







        height: 100%;







        border-radius: 8px;







        background: #75a8ca;







    }























    /* =====================================================







       SESSION STATISTICS







       ===================================================== */















    .nwd-stats-card {







        background: rgba(255, 255, 255, 0.90);







        border: 1px solid #e5e5e2;







        border-radius: 20px;







        padding: 13px;







        margin-top: 0;







        box-sizing: border-box;







    }















    .nwd-stats-title {







        display: flex;







        align-items: center;







        justify-content: space-between;







        color: #27313b;







        font-size: 16px;







        font-weight: 700;







        margin-bottom: 10px;







    }















    .nwd-duration {







        color: #59636c;







        font-size: 13px;







        font-weight: 500;







    }















    .nwd-stat-grid {







        display: grid;







        grid-template-columns: repeat(4, 1fr);







        gap: 8px;







    }















    .nwd-stat-box {







        background: #faf9f6;







        border-radius: 15px;







        min-height: 90px;







        padding: 10px;







        text-align: center;







        box-sizing: border-box;







    }















    .nwd-stat-icon {







        font-size: 18px;







        margin-bottom: 5px;







    }















    .nwd-stat-label {







        color: #69727a;







        font-size: 12px;







        font-weight: 600;







    }















    .nwd-stat-number {







        color: #27313b;







        font-size: 22px;







        font-weight: 750;







        margin-top: 7px;







    }















    .nwd-stat-percent {







        color: #69727a;







        font-size: 12px;







    }























    /* =====================================================







       STREAMLIT BUTTONS







       ===================================================== */















    div.stButton> button {







        width: 100%;







        height: 55px;







        border-radius: 28px;







        border: none;







        font-size: 15px;







        font-weight: 650;







        transition: all 0.2s ease;







    }















    div.stButton> button:hover {







        transform: translateY(-1px);







        box-shadow: 0 4px 12px rgba(70, 90, 110, 0.15);







    }















    @media (max-width: 1050px) {







        .nwd-header-right {







            width: 430px;







        }















        .nwd-road {







            width: 650px;







            right: -190px;







            height: 126px;







        }















        .nwd-slogan {







            margin-right: 220px;







            font-size: 14px;







        }















        .nwd-title {







            font-size: 36px;







        }















        .nwd-subtitle {







            font-size: 17px;







        }















        .nwd-brand-text {







            min-width: 0;







            width: auto;







            flex: 1 1 auto;







            margin-left: 0;







        }















        .nwd-title {







            transform: scaleX(1.12);







        }















        .nwd-subtitle {







            transform: scaleX(1.12);







        }















        .nwd-logo {







            width: 112px;







            height: 112px;







        }







    }















</style>







""")























# =========================================================*







# HEADER*







# =========================================================*















render_html(f"""







<div class="nwd-header">















    <div class="nwd-brand">















        <img







            class="nwd-logo"







            src="{LOGO_DATA_URI}"







            alt="No Way Drowsy Logo"







       >















        <div class="nwd-brand-text">







            <div class="nwd-title">No Way Drowsy</div>







            <div class="nwd-subtitle">







                Driver Drowsiness Detection System







            </div>







        </div>















    </div>















    <div class="nwd-header-right">







        <img







            class="nwd-road"







            src="{ROAD_DATA_URI}"







            alt="Road illustration"







       >















        <div class="nwd-slogan">







            <div>







                Stay Alert<br>







                Reach Safer







            </div>







            <div class="nwd-slogan-line"></div>







        </div>







    </div>















</div>







""")























# =========================================================*







# MAIN LAYOUT*







# =========================================================*















left, right = st.columns([1.06, 0.94], gap="medium")























# =========================================================*







# LEFT — LIVE CAMERA*







# =========================================================*















with left:







    if "camera_running" not in st.session_state:



        st.session_state.camera_running = False







    # Camera card header — same visual structure as the original.*

    render_html("""

    <div style="

        background:#ffffff;

        border-radius:22px 22px 0 0;

        box-shadow:0 6px 18px rgba(50,60,70,0.10);

        height:76px;

        display:flex;

        align-items:center;

        justify-content:space-between;

        padding:0 22px;

        box-sizing:border-box;

        margin:0;

    ">

        <div style="

            display:flex;

            align-items:center;

            gap:10px;

            font-size:16px;

            font-weight:750;

            color:#27313b;

        ">

            📹 <span>Live Camera</span>

        </div>

        <div style="

            display:flex;

            align-items:center;

            gap:8px;

            padding:10px 16px;

            border:1px solid #ecebe8;

            border-radius:24px;

            color:#68727b;

            font-size:14px;

            background:#fff;

        ">

            <span style="

                width:10px;

                height:10px;

                border-radius:50%;

                background:#49b879;

                display:inline-block;

            "></span>

            Detecting...

        </div>

    </div>

    """)



    live_camera()



    b1, b2, b3 = st.columns(3, gap="small")



    with b1:

        if st.button("▶ Start Monitoring", use_container_width=True):

            st.session_state.camera_running = True

            st.session_state.camera_frame_count = 0

            st.session_state.last_yolo_frame = None

            st.rerun()



    with b2:

        if st.button("■ Stop Monitoring", use_container_width=True):

            st.session_state.camera_running = False

            close_camera()

            st.session_state.camera_frame_count = 0

            st.session_state.last_yolo_frame = None

            st.rerun()





    with b3:

        if st.button("↻ Reset Session", use_container_width=True):
            st.session_state.camera_running = False
            close_camera()

            st.session_state.camera_frame_count = 0
            st.session_state.last_yolo_frame = None

            st.session_state.eye_state = "NO DETECTION"
            st.session_state.perclos = 0.0
            st.session_state.closed_streak = 0
            st.session_state.fps = 0
            st.session_state.alert_status = "READY"
            st.session_state.alert_active = False

            st.rerun()





# RIGHT — DRIVER STATUS*







# =========================================================*















with right:















    render_html("""







    <div class="nwd-card nwd-status-card">















        <div class="nwd-status-header">







            👤







            <span>Driver Status</span>







        </div>















        <div class="nwd-status-body">















            <div class="nwd-status-circle"></div>















            <div>







                <div class="nwd-status-main">NORMAL</div>







                <div class="nwd-status-sub">







                    Driver is alert







                </div>







            </div>















        </div>















    </div>







    """)
    # -----------------------------------------------------*
    # EYE STATE + PERCLOS*

    # -----------------------------------------------------*
    c1, c2 = st.columns(2, gap="small")

    with c1:

        components.html(
            """
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: transparent;
                }

                .nwd-metric-card {
                    background: #edf2f7;
                    border: 1px solid #e5e9ed;
                    border-radius: 16px;
                    min-height: 96px;
                    height: 96px;
                    padding: 11px 15px;
                    margin: 0;
                    box-sizing: border-box;
                }

                .nwd-metric-label {
                    color: #56616b;
                    font-size: 14px;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .nwd-metric-value {
                    color: #27313b;
                    font-size: 27px;
                    font-weight: 750;
                    margin-top: 6px;
                }
            </style>

            <div class="nwd-metric-card">
                <div class="nwd-metric-label">
                    👁
                    <span>Eye State</span>
                </div>

                <div id="eye-state-value" class="nwd-metric-value">
                    OPEN
                </div>
            </div>

            <script>
                const eyeStateValue =
                    document.getElementById("eye-state-value");

                async function updateEyeState() {
                    try {
                        const response = await fetch(
                            "http://127.0.0.1:8765/latest"
                        );

                        if (!response.ok) {
                            throw new Error(
                                "HTTP " + response.status
                            );
                        }

                        const data = await response.json();

                        const state = (
                            data.eye_state || "NO DETECTION"
                        ).toUpperCase();

                        if (eyeStateValue) {
                            eyeStateValue.textContent = state;
                        }
                    } catch (error) {
                        console.error(
                            "[No Way Drowsy] Eye State error:",
                            error
                        );
                    }
                }

                updateEyeState();
                setInterval(updateEyeState, 500);
            </script>
            """,
            height=96,
            scrolling=False,
        )

    with c2:
        components.html(
            """
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: transparent;
                }

                .nwd-metric-card {
                    background: #edf2f7;
                    border: 1px solid #e5e9ed;
                    border-radius: 16px;
                    min-height: 96px;
                    height: 96px;
                    padding: 11px 15px;
                    margin: 0;
                    box-sizing: border-box;
                }

                .nwd-metric-label {
                    color: #56616b;
                    font-size: 14px;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .nwd-metric-value {
                    color: #27313b;
                    font-size: 27px;
                    font-weight: 750;
                    margin-top: 6px;
                }

                .nwd-progress-bg {
                    width: 100%;
                    height: 8px;
                    border-radius: 8px;
                    background: #d9dde1;
                    margin-top: 6px;
                    overflow: hidden;
                }

                .nwd-progress-fill {
                    height: 100%;
                    border-radius: 8px;
                    background: #75a8ca;
                }
            </style>

            <div class="nwd-metric-card">

                <div class="nwd-metric-label">
                    📊
                    <span>PERCLOS</span>
                </div>

                <div
                    id="perclos-value"
                    class="nwd-metric-value">
                    0.00
                </div>

                <div class="nwd-progress-bg">
                    <div
                        id="perclos-fill"
                        class="nwd-progress-fill"
                        style="width: 0%;">
                    </div>
                </div>

            </div>

            <script>
                const perclosValue =
                    document.getElementById("perclos-value");

                const perclosFill =
                    document.getElementById("perclos-fill");

                const eyeSamples = [];

                const MAX_SAMPLES = 90;

                async function updatePerclos() {
                    try {
                        const response = await fetch(
                            "http://127.0.0.1:8765/latest"
                        );

                        if (!response.ok) {
                            throw new Error(
                                "HTTP " + response.status
                            );
                        }

                        const data = await response.json();

                        const state = (
                            data.eye_state || "NO DETECTION"
                        ).toUpperCase();

                        if (
                            state === "OPEN" ||
                            state === "CLOSED"
                        ) {
                            eyeSamples.push(
                                state === "CLOSED" ? 1 : 0
                            );

                            if (
                                eyeSamples.length >
                                MAX_SAMPLES
                            ) {
                                eyeSamples.shift();
                            }
                        }

                        if (eyeSamples.length > 0) {

                            const closedSamples =
                                eyeSamples.reduce(
                                    (sum, value) =>
                                        sum + value,
                                    0
                                );

                            const perclos =
                                closedSamples /
                                eyeSamples.length;

                            if (perclosValue) {
                                perclosValue.textContent =
                                    perclos.toFixed(2);
                            }

                            if (perclosFill) {
                                const progress =
                                    Math.min(
                                        (perclos / 0.70) * 100,
                                        100
                                    );

                                perclosFill.style.width =
                                    progress + "%";
                            }
                        }

                    } catch (error) {
                        console.error(
                            "[No Way Drowsy] PERCLOS error:",
                            error
                        );
                    }
                }

                updatePerclos();
                setInterval(updatePerclos, 500);
            </script>
            """,
            height=96,
            scrolling=False,
        )
    # -----------------------------------------------------*
    # CLOSED STREAK + FPS + ALERT*
    # -----------------------------------------------------*

    c1, c2, c3 = st.columns(
        [1.15, 0.85, 1.0],

        gap="small"
    )

    with c1:
        components.html(
            """
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: transparent;
                }

                .nwd-metric-card {
                    background: #edf2f7;
                    border: 1px solid #e5e9ed;
                    border-radius: 16px;
                    min-height: 96px;
                    height: 96px;
                    padding: 11px 15px;
                    margin: 0;
                    box-sizing: border-box;
                }

                .nwd-metric-label {
                    color: #56616b;
                    font-size: 14px;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .nwd-metric-value {
                    color: #27313b;
                    font-size: 27px;
                    font-weight: 750;
                    margin-top: 6px;
                }

                .nwd-metric-small {
                    color: #6d757d;
                    font-size: 12px;
                }

                .nwd-progress-bg {
                    width: 100%;
                    height: 8px;
                    border-radius: 8px;
                    background: #d9dde1;
                    margin-top: 6px;
                    overflow: hidden;
                }

                .nwd-progress-fill {
                    height: 100%;
                    border-radius: 8px;
                    background: #75a8ca;
                }
            </style>

            <div class="nwd-metric-card">

                <div class="nwd-metric-label">
                    ▱
                    <span>Closed Streak</span>
                </div>

                <div class="nwd-metric-value">
                    <span id="closed-streak-value">0</span>
                    <span class="nwd-metric-small">/ 20</span>
                </div>

                <div class="nwd-progress-bg">
                    <div
                        id="closed-streak-fill"
                        class="nwd-progress-fill"
                        style="width: 0%;">
                    </div>
                </div>

            </div>

            <script>
                let closedStreak = 0;

                const closedStreakValue =
                    document.getElementById("closed-streak-value");

                const closedStreakFill =
                    document.getElementById("closed-streak-fill");

                async function updateClosedStreak() {
                    try {
                        const response = await fetch(
                            "http://127.0.0.1:8765/latest"
                        );

                        if (!response.ok) {
                            throw new Error(
                                "HTTP " + response.status
                            );
                        }

                        const data = await response.json();

                        const state = (
                            data.eye_state || "NO DETECTION"
                        ).toUpperCase();

                        if (state === "CLOSED") {
                            closedStreak = Math.min(
                                closedStreak + 1,
                                20
                            );
                        } else if (state === "OPEN") {
                            closedStreak = 0;
                        }

                        if (closedStreakValue) {
                            closedStreakValue.textContent =
                                closedStreak;
                        }

                        if (closedStreakFill) {
                            closedStreakFill.style.width =
                                ((closedStreak / 20) * 100) + "%";
                        }

                    } catch (error) {
                        console.error(
                            "[No Way Drowsy] Closed Streak error:",
                            error
                        );
                    }
                }

                updateClosedStreak();
                setInterval(updateClosedStreak, 500);
            </script>
            """,
            height=96,
            scrolling=False,
        )

    with c2:
        components.html(
            """
            <style>
                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: transparent;
                }

                .nwd-metric-card {
                    background: #edf2f7;
                    border: 1px solid #e5e9ed;
                    border-radius: 16px;
                    min-height: 96px;
                    height: 96px;
                    padding: 11px 15px;
                    margin: 0;
                    box-sizing: border-box;
                }

                .nwd-metric-label {
                    color: #56616b;
                    font-size: 14px;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .nwd-metric-value {
                    color: #27313b;
                    font-size: 27px;
                    font-weight: 750;
                    margin-top: 6px;
                }
            </style>

            <div class="nwd-metric-card">

                <div class="nwd-metric-label">
                    ⚡
                    <span>FPS</span>
                </div>

                <div
                    id="fps-value"
                    class="nwd-metric-value">
                    0
                </div>

            </div>

            <script>
                const fpsValue =
                    document.getElementById("fps-value");

                let frameTimes = [];

                async function updateFPS() {
                    try {
                        const startTime = performance.now();

                        const response = await fetch(
                            "http://127.0.0.1:8765/latest"
                        );

                        if (!response.ok) {
                            throw new Error(
                                "HTTP " + response.status
                            );
                        }

                        await response.json();

                        const currentTime =
                            performance.now();

                        frameTimes.push(currentTime);

                        frameTimes =
                            frameTimes.filter(
                                time =>
                                    currentTime - time <= 1000
                            );

                        const fps =
                            frameTimes.length;

                        if (fpsValue) {
                            fpsValue.textContent = fps;
                        }

                    } catch (error) {
                        console.error(
                            "[No Way Drowsy] FPS error:",
                            error
                        );
                    }
                }

                updateFPS();
                setInterval(updateFPS, 100);
            </script>
            """,
            height=96,
            scrolling=False,
        )

with c3:
    render_html("""
    <div class="nwd-metric-card">

        <div class="nwd-metric-label">

            🔔
            <span>Alert Status</span>

        </div>

        <div
            class="nwd-metric-value"

            style="font-size: 20px;">
            Active 🔊
        </div>
    </div>
    """)
    render_html(f"""
<div class="nwd-bottom-road">
    <img
        src="{ROAD_BG_DATA_URI}"
        alt="Road background"
    >
</div>
""")