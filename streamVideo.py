#########################################################################################################
#
# DBLab
# Machine Learning Utils
# 
# >> streamlit run streamvideo.py
# Stream de Vídeo para Visão Computacional no Streamlit.
# Créditos: https://github.com/whitphx/streamlit-webrtc/
# Azzi - 06/2021
#
#########################################################################################################

import av
import cv2
import numpy as np
import streamlit as st
import imutils

from streamlit_webrtc import (
	AudioProcessorBase,
	ClientSettings,
	VideoProcessorBase,
	WebRtcMode,
	webrtc_streamer,
)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
	rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
	media_stream_constraints={
		"video": True,
		"audio": False,
	},
)

#################################################################################
# Modelo:
# Crie uma função pro seu modelo a partir do input capturado pela webcam (img)
#################################################################################
def modelo(img):

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

		## code

		## Formato da imagem: ndarray (uint-8)

		return img # Retorna a imagem manipulada

# Processamento do vídeo
class VideoProcessor(VideoProcessorBase):
	confidence_threshold: float
	result_queue: "queue.Queue[List[Detection]]"

	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
		image = frame.to_ndarray(format="bgr24")
		
		# Formato da imagem: ndarray (uint-8)

		# Modelo: mudar função!
		annotated_image = modelo(image) ## <-----------------------------------------------------

		return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# Chama o stream de vídeo WebRTC
webrtc_ctx = webrtc_streamer(
	key="object-detection",
	mode=WebRtcMode.SENDRECV,
	client_settings=WEBRTC_CLIENT_SETTINGS,
	video_processor_factory=VideoProcessor,
	async_processing=True,
)
