from streamlit_webrtc import ClientSettings
import os

CLASSES = open('YOLO/classes.names').read().strip().split('\n')



WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )