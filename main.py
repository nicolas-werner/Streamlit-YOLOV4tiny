import asyncio
import base64
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
from PIL import Image
try:
	from typing import Literal
except ImportError:
	from typing_extensions import Literal 

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np

import streamlit as st
from aiortc.contrib.media import MediaPlayer

import time
import pandas as pd



from streamlit_webrtc import (
	AudioProcessorBase,
	ClientSettings,
	VideoProcessorBase,
	WebRtcMode,
	webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


st.set_page_config(page_title="Objekterkennung mit Yolov4-tiny", page_icon="🐢")


WEBRTC_CLIENT_SETTINGS = ClientSettings(
	rtc_configuration={"iceServers": [
		# aus https://github.com/andBabaev/streamlit_yolov5/blob/master/config.py und https://github.com/whitphx/streamlit-webrtc/blob/main/app.py -scheint ein Standard zu sein. Auf jeden Fall klappt das.
		{"urls": ["stun:stun.l.google.com:19302"]}]},
	media_stream_constraints={
		"video": True,
		"audio": False,
	},
)


def main():

	st.title("Objekterkennung mit YOLO")
	object_detection_page = "Live-Erkennung von Statuen"
	image_detection_page = "Erkennen von Statuen auf Bildern"
	info_page = "Informationen zum Projekt"


	app_mode = st.sidebar.selectbox(
		"Modus wählen:",
		[
			object_detection_page,
			image_detection_page,
			info_page
		],
	)
	st.subheader(app_mode)
	if app_mode == object_detection_page:
		object_detection()
	elif app_mode == image_detection_page:
		image_detection()

	elif app_mode == info_page:
		st.markdown("Das Projekt ist im Rahmen der Lehrveranstaltung [Computer Vision und kulturelles Erbe](https://lehre.idh.uni-koeln.de/lehrveranstaltungen/sosem21/computer-vision-und-kulturelles-erbe/) der Universität zu Köln entstanden. Grundlage für die Webanwendung bildet ein Convolutional Neural Network, das darauf trainiert wurde, Statuen und Denkmäler in Köln zu erkennen.")
		st.subheader("Trainieren eines YOLOV4-tiny Modells")
		st.markdown("#### Das Datenset")
		st.markdown('Mithilfe der Photogrammetry-App [Trnio](https://www.trnio.com/) haben wir einige Statuen und Denkmäler in Köln eingescannt und mit dem Unity Package [Perception](https://github.com/Unity-Technologies/com.unity.perception) ein synthetisches Datenset erzeugt. Dazu wurden die Statuen vor einen Hintergrund platziert, wobei nach jedem Frame sich der Hintergrund, Lichtintensivität sowie der Rotationswinkel der einzelnen Statuen auf der Z-Achse verändert hat, sodass das Modell lernt, die Statuen in möglichst vielen Situationen zu erkennen.  Insgesamt bestand unser Datenset aus 16000 annotierten Einzelbildern, wobei sich die die Annotation pro Statue auf etwa 9000 belief.')
		bilduni=Image.open('Bilder/unity1.png')
		st.image(bilduni, caption="Erzeugen eines synthetischen Datensets mit dem Unity Package Perception")
		
		st.text('')
		bild=Image.open('Bilder/datenset.png')
		st.image(bild, caption="Größe des synthetischen Datensets", width=700)
		st.markdown("#### Transfer Learning mit Google Colab und Darknet")
		st.markdown("Das Modell wurde mit dem Framework [Darknet](https://pjreddie.com/darknet/) und mit einem bereits vortrainierten Objekterkennungsmodell auf der Plattform Google Colab per Transfer Learning trainert. Das Notebook kann [hier](https://colab.research.google.com/drive/1IvQiI_iVTBGzdsJAjLgkP0xMWdjnpAVp?usp=sharing) eingesehen werden.")
		bild2=Image.open('Bilder/loss.png')
		bild3=Image.open('Bilder/testset.png')
		bildlist = [bild2, bild3]
		caption_list=["Mean Average Precision und Loss", " Statuenerkennung: Beispiel aus dem Test-Set"]
		st.image(bildlist, caption=caption_list, width=349)
		st.subheader('Demo')
		# Das Modell kann nicht nur die Statuen in freier Wildbahn erkennen, sondern auch auf Bildern
		video1 = open('Videos/video.mp4', 'rb')
		video_file1 = video1.read()
		st.video(video_file1)
		st.markdown('Das Modell kann auch die Statuen auf Bildern auf dem Smartphone erkennen')
		video = open('Videos/ddemo.mov', 'rb')
		video_file = video.read()
		st.video(video_file)
		bild99=Image.open('Bilder/testset2.png')
		st.image(bild99, caption='weitere Bilder aus dem Test-Set')
		bild91=Image.open('Bilder/real.png')
		st.image(bild91, caption='Erkennen von Statuen in der Realität')
	   

		#https://colab.research.google.com/drive/1IvQiI_iVTBGzdsJAjLgkP0xMWdjnpAVp?usp=sharing
	# COCO hinzufügen?

	# option = st.selectbox(
	#     'Please Select the Configuration file', ("yolov4-tiny.cfg",), )

	# option = st.selectbox('Please Select the Weight file',
	#                       ("yolov4-tiny.weights",))

 

	#st.info("es kann etwas dauern, bis die Webcam geladen hat. Test: zwischen 30-60 Sekunden")



	

# Threshold für OpenCV
Conf_threshold = 0.80
NMS_threshold = 0.2

# Colours
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
		  (255, 255, 0), (255, 0, 255), (0, 255, 255)]




STATUEN = "YOLO/classes.names"


class_name = []
with open(STATUEN, 'rt') as f:
	class_name = f.read().rstrip('\n').split('\n')

model_config_file = "YOLO/yolov4-tiny-custom.cfg"
model_weight = "YOLO/yolov4-tiny-custom_final.weights"


# darknet mit DNN einlesen
net = cv2.dnn.readNet(model_weight, model_config_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load Model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# quelle https://github.com/whitphx/streamlit-webrtc und https://github.com/rahularepaka/ObjectDetectionYolov4Web
def object_detection():

	class Video(VideoProcessorBase):

		def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
			image = frame.to_ndarray(format="bgr24")

			classes, scores, boxes = model.detect(
				image, Conf_threshold, NMS_threshold)
			for (classid, score, box) in zip(classes, scores, boxes):

				color = COLORS[int(classid) % len(COLORS)]

				label = "%s : %f" % (class_name[classid[0]], score)

				cv2.rectangle(image, box, color, 1)
				cv2.putText(image, label, (box[0], box[1]-10),
							cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

			return av.VideoFrame.from_ndarray(image, format="bgr24")

	webrtc_ctx = webrtc_streamer(
		key="object-detection",
		mode=WebRtcMode.SENDRECV,
		client_settings=WEBRTC_CLIENT_SETTINGS,
		video_processor_factory=Video,
		async_processing=True,
	)


# Bilder hochladen


def image_detection():
	newimgs = st.file_uploader ("Choose images to upload", accept_multiple_files=True)
	if newimgs is not None:
		for newimg in newimgs:
			label = "There is no statue from our Database in this picture! :("
			dim = (416, 416)
			file_bytes = np.asarray(bytearray(newimg.read()), dtype=np.uint8)
			uploadedimg = cv2.imdecode(file_bytes, 1)
			classes, scores, boxes = model.detect(uploadedimg, Conf_threshold, NMS_threshold)
			for (classid, score, box) in zip(classes, scores, boxes):
				color = COLORS[int(classid) % len(COLORS)]
				label = "%s : %f" % (class_name[classid[0]], score)
				cv2.rectangle(uploadedimg, box, color, 1)
				cv2.putText(uploadedimg, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
			
			uploadedimg = cv2.cvtColor(uploadedimg, cv2.COLOR_BGR2RGB)
			st.image(uploadedimg, caption = label)


if __name__ == "__main__":

	main()