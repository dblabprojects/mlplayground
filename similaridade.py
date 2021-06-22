#########################################################################################################
#
# DBLab
# POC - Busca por Similaridade
# 
# >> streamlit run similaridade.py
# Extração de features de fotos e busca por similaridade
# Flávio, Masiero, Joice, Azzi - 06/2021
#
#########################################################################################################

import streamlit as st
import numpy as np
from keras.preprocessing import image
import os
import keras
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from time import time
from scipy.spatial import distance
from joblib import dump, load
import cv2
from numpy import asarray
from PIL import Image,ImageEnhance

# Transforma imagens em um arrays compatíveis com o modelo da VGG (224x224x3)
def load_image(img, model):
    #img = asarray(img)
    #img = image.load_img(path, target_size=model.input_shape[1:3])
    
    img = np.array(img)
    img = cv2.resize(img, (model.input_shape[1:3]))
    print(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# Calcula a distância entre cossenos dos vetores de features e retorna o index das imagens com menores distâncias
def get_closest_images(pca_feat, pca_features, num_results=10):
    distances=[]
    for i in range(len(pca_features)):
        distances.append((i, distance.cosine(pca_feat, pca_features[i])))
    idx_closest = sorted(distances, key=lambda d: d[1])[0:num_results]
    return idx_closest

def format_image(path, thumb_height):
    img = image.load_img(path)
    img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
    return img

# Redimensiona e concatena as imagens similares
def get_concatenated_images(indexes, thumb_height, path_images):
    thumbs = []
    print(path_images)
    for index in indexes:
        idx = index[0]
        print(path_images[idx])
        img = format_image(path_images[idx], 200)
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

# Extrai as features das fotos de input a partir do index e realiza a busca das imagens similares
def busca(images, num_res):
    dataset = init()

    # Rede VGG16
    model = keras.applications.VGG16(weights='imagenet', include_top=True)

    tic = time()

    img, x = load_image(images, model)

    # Extração de features
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)   
    feat = feat_extractor.predict(x)[0]

    # load PCA features
    pca_features = np.load('./pca_features_netshoes_'+ str(len(dataset)) +'.npy')
    # load PCA model
    pca = load(open('./pcaModel_netshoes_'+ str(len(dataset)) +'.joblib', 'rb'))

    # Principal Component Analysis (PCA)
    pca_feat = pca.transform([feat])

    # Busca de index das imagens similares
    idx_closest = get_closest_images(pca_feat, pca_features, num_res)
    print(idx_closest)

    #query_image = format_image(images, 300) # Fotos de input
    # query_image = format_image(dataset_images[query_image_idx], 300)
    results_image = get_concatenated_images(idx_closest, 200, dataset) # Output da busca

    toc = time()
    elap = toc-tic
    print("analyzing image. Time: %4.4f seconds." % (elap))

    return results_image, feat

@st.cache
def init():
	## Inicialização e setup ##

	tic = time()

	# Rede VGG16
	model = keras.applications.VGG16(weights='imagenet', include_top=True)
	    
	feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
	# feat_extractor.summary()

	pca = PCA(n_components=300)

	# Caminho do diretório com imagens de busca
	dataset_images_path = './augmented/batch_1/'

	image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)

	dataset_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]

	tac = time()
	elap = tac-tic
	print('finished loading images. Time: %4.4f seconds.' % elap)

	tic = time()

	# load PCA features
	pca_features = np.load('./pca_features_netshoes_'+ str(len(dataset_images)) +'.npy')
	# load PCA model
	pca = load(open('./pcaModel_netshoes_'+ str(len(dataset_images)) +'.joblib', 'rb'))

	tac = time()
	elap = tac-tic
	print('finished loading models. Time: %4.4f seconds.' % elap)	

	return dataset_images

def main():

	st.subheader("Busca por Tênis 👟")

	num_obj = st.sidebar.slider(
		label = "Buscar por quantos objetos?", 
		min_value = 1,
		max_value = 10,
		value = 5,
		help = "Selecione o número de objetos a serem buscados.")

	uploaded = False

	while uploaded is False:
		uploaded_image = st.file_uploader("Faça upload de uma imagem (.jpg, .jpeg, .png):", type=['jpg', 'png', 'jpeg'])

		if uploaded_image is not None:
			image = Image.open(uploaded_image)
			uploaded = True
			st.image(image)
		
	if st.button('Iniciar busca'):

		with st.spinner('Buscando...'):
			tic = time()

			result, features = busca(image, num_obj)

			tac = time()
			elap = tac-tic

		st.text('Tênis similares encontrados:')
		st.image(result)

		st.text('Busca finalizada. Tempo: %4.4f segundos.' % elap)
		print('Busca finalizada. Tempo: %4.4f segundos.' % elap)
		st.balloons()

		st.sidebar.text('Features extraídas:')
		st.sidebar.line_chart(data = features, height=100)

if __name__ == "__main__":
    main()



