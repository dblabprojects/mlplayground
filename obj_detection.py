#########################################################################################################
#
# DBLab
# Machine Learning Playground
# 
# >> streamlit run home.py
# Detecção de Objetos com MobileNet + Single Shot Detector (SSD) + Deep Neural Network (DNN)
# https://github.com/robmarkcole/object-detection-app
# Azzi - 06/2021
#
#########################################################################################################

import queue
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

try:
	from typing import Literal
except ImportError:
	from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
from aiortc.contrib.media import MediaPlayer
import streamlit as st

from streamlit_webrtc import (
	AudioProcessorBase,
	ClientSettings,
	VideoProcessorBase,
	WebRtcMode,
	webrtc_streamer,
)

from imutils.video import FPS

import pafy
from io import StringIO

import tempfile

HERE = Path(__file__).parent

WEBRTC_CLIENT_SETTINGS = ClientSettings(
	rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
	media_stream_constraints={
		"video": True,
		"audio": False,
	},
)

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
	# Don't download the file twice.
	# (If possible, verify the download using the file length.)
	if download_to.exists():
		if expected_size:
			if download_to.stat().st_size == expected_size:
				return
		else:
			st.info(f"{url} is already downloaded.")
			if not st.button("Download again?"):
				return

	download_to.parent.mkdir(parents=True, exist_ok=True)

	# These are handles to two visual elements to animate.
	weights_warning, progress_bar = None, None
	try:
		weights_warning = st.warning("Downloading %s..." % url)
		progress_bar = st.progress(0)
		with open(download_to, "wb") as output_file:
			with urllib.request.urlopen(url) as response:
				length = int(response.info()["Content-Length"])
				counter = 0.0
				MEGABYTES = 2.0 ** 20.0
				while True:
					data = response.read(8192)
					if not data:
						break
					counter += len(data)
					output_file.write(data)

					# We perform animation by overwriting the elements.
					weights_warning.warning(
						"Downloading %s... (%6.2f/%6.2f MB)"
						% (url, counter / MEGABYTES, length / MEGABYTES)
					)
					progress_bar.progress(min(counter / length, 1.0))
	# Finally, we remove these visual elements by calling .empty().
	finally:
		if weights_warning is not None:
			weights_warning.empty()
		if progress_bar is not None:
			progress_bar.empty()

def mobilessd(image):

	fps = FPS().start()

	"""Object detection demo with MobileNet SSD.
	This model and code are based on
	https://github.com/robmarkcole/object-detection-app
	"""
	MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
	MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
	PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
	PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

	CLASSES = [
		"background",
		"aeroplane",
		"bicycle",
		"bird",
		"boat",
		"bottle",
		"bus",
		"car",
		"cat",
		"chair",
		"cow",
		"diningtable",
		"dog",
		"horse",
		"motorbike",
		"person",
		"pottedplant",
		"sheep",
		"sofa",
		"train",
		"tvmonitor",
	]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
	download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

	DEFAULT_CONFIDENCE_THRESHOLD = 0.5

	class Detection(NamedTuple):
		name: str
		prob: float

	VideoProcessorBase._net = cv2.dnn.readNetFromCaffe(
		str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
	)

	blob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
		)
	VideoProcessorBase._net.setInput(blob)
	detections = VideoProcessorBase._net.forward()


	# Exibe marca
	cv2.putText(
		image,
		"DBLab",
		(50, 50),
		cv2.FONT_HERSHEY_SIMPLEX,
		1,
		(255, 0, 0),
		2
	)

	# loop over the detections
	(h, w) = image.shape[:2]
	result: List[Detection] = []
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			name = CLASSES[idx]
			result.append(Detection(name=name, prob=float(confidence)))

			# display the prediction
			label = f"{name}: {round(confidence * 100, 2)}%"
			cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(
				image,
				label,
				(startX, y),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				COLORS[idx],
				2,
			)
			fps.update()

	fps.stop()
	
	frames = str("{:.2f}".format(fps.fps())) + ' FPS'

	# Exibe FPS
	cv2.putText(
		image,
		frames,
		(50, 450),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.5,
		(255, 0, 0),
		2
	)

	return image

def yolov3(img):

	fps = FPS().start()

	# Load Yolo
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	# Exibe marca
	cv2.putText(
		img,
		"DBLab",
		(50, 50),
		cv2.FONT_HERSHEY_SIMPLEX,
		1,
		(255, 0, 0),
		2
	)

	while True: ## Leitura de Frames
		height, width, channels = img.shape

		# Detecting objects
		blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)

		# Showing informations on the screen
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					# Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)
					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)
					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

		font = cv2.FONT_HERSHEY_PLAIN
		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				label = str(classes[class_ids[i]])
				color = colors[i]
				cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
				cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
				fps.update()
		## Formato da imagem: ndarray (uint-8)

		fps.stop()
		
		frames = str("{:.2f}".format(fps.fps())) + ' FPS'

		# Exibe FPS
		cv2.putText(
			img,
			frames,
			(50, 450),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 0, 0),
			2
		)

		return img # Retorna a imagem manipulada

def videoStream(choice):
	st.text('Selecione uma câmera e clique em Start.')
	# Processamento do vídeo
	class VideoProcessor(VideoProcessorBase):
		confidence_threshold: float
		result_queue: "queue.Queue[List[Detection]]"

		def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
			image = frame.to_ndarray(format="bgr24")

			# Formato da imagem: ndarray (uint-8)
			if choice == 'YOLO v3':
				annotated_image = yolov3(image)
			if choice == 'MobileNet SSD':
				annotated_image = mobilessd(image)

			return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

	# Chama o stream de vídeo WebRTC
	webrtc_ctx = webrtc_streamer(
		key=str(choice),
		mode=WebRtcMode.SENDRECV,
		client_settings=WEBRTC_CLIENT_SETTINGS,
		video_processor_factory=VideoProcessor,
		async_processing=True,
	)

def youtube(url, model):
	video = pafy.new(url)
	best = video.getbest(preftype="mp4")

	capture = cv2.VideoCapture(best.url)

	frames = []
	stframe = st.empty()
	
	while (capture.isOpened()):
		succ, frame = capture.read()
		if succ:
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			if model == 'MobileNet SSD':
				annotated_frame = mobilessd(frame_rgb)
			elif model == 'YOLO v3':
				annotated_frame = yolov3(frame_rgb)

			frames.append(annotated_frame)
			stframe.image(annotated_frame)

		else:
			break

def upload(model):
	f = st.file_uploader("Choose a file")

	if f is not None:
		tfile = tempfile.NamedTemporaryFile(delete=False) 
		tfile.write(f.read())

		vf = cv2.VideoCapture(tfile.name)

		stframe = st.empty()

		while vf.isOpened():
			ret, frame = vf.read()
			# if frame is read correctly ret is True
			if not ret:
				print("Can't receive frame (stream end?). Exiting ...")
				break
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			if model == 'MobileNet SSD':
				annotated_frame = mobilessd(frame_rgb)
			elif model == 'YOLO v3':
				annotated_frame = yolov3(frame_rgb)

			stframe.image(annotated_frame)


def main():
	st.header("**Visão Computacional:** Detecção de Objetos")

	models = [	"YOLO v3", "MobileNet SSD" ]

	model_choice = st.sidebar.selectbox("Modelo:", models)

	modes = [	"Webcam", "YouTube", "Upload" ]

	mode_choice = st.sidebar.selectbox("Modo:", modes)

	if model_choice == 'YOLO v3':
		if mode_choice == 'Webcam':
			videoStream(model_choice)
		elif mode_choice == 'YouTube':
			link = st.text_input('Link do YouTube', 'https://www.youtube.com/watch?v=Ri0VbeNUGhg')
			if link != '': youtube(link, model_choice)	

		st.markdown("O **YOLO**: **Y**ou **O**nly **L**ook **O**nce é uma ferramenta para detecção e classificação de objetos em tempo real que, em uma pequena fração de segundo - dez vezes mais rápido que um piscar de olhos - consegue detectar até 80 classes de objetos diferentes em uma imagem.")
		st.markdown("Representa o estado da arte em sistemas de reconhecimento de objetos em tempo real, de acordo com um compromisso entre velocidade e assertividade. Também é totalmente código aberto e livre de licenças de uso. Ou seja, tudo nesta tecnologia (o código-fonte, a arquitetura da rede neural, os pesos com as quais esta rede é executada e os datasets utilizados para treinar) é livre e pode ser usado por qualquer um, de qualquer forma.")
		st.markdown("Quer saber mais? Leia nosso [artigo no Medium](https://medium.com/@dblab/yolo-um-sistema-para-detec%C3%A7%C3%A3o-de-classes-de-objetos-em-tempo-real-be94c790c3e8) e chama a gente! [dblab@dbserver.com.br](mailto:dblab@dbserver.com.br)")
	
	if model_choice == 'MobileNet SSD':
		if mode_choice == 'Webcam':
			videoStream(model_choice)
		elif mode_choice == 'YouTube':
			link = st.text_input('Link do YouTube', 'https://www.youtube.com/watch?v=Ri0VbeNUGhg')
			if link != '': youtube(link, model_choice)
		elif mode_choice == 'Upload':
			upload(model_choice)


		st.markdown("A **MobileNet** é uma classe de convolução de Redes Neurais que simplifica a criação de aplicações para reconhecimento de imagens em dispositivos móveis e na web. Por padrão, a rede utiliza o dataset ImageNet da Google, que contém um banco com mais de 1.500.00 imagens classificadas em 1.000 categorias. A precisão da MobileNet é menor do que as redes neurais tradicionais, mas compensa na velocidade de processamento e a grande quantidade de amostras disponíveis.")
		st.markdown("Quer saber mais? Chama a gente! [dblab@dbserver.com.br](mailto:dblab@dbserver.com.br)")
		
		st.markdown(
		"Créditos: @robmarkcole, @whitphx."
		)

if __name__ == "__main__":
	main()