#########################################################################################################
#
# DBLab
# Machine Learning Playground
# 
# >> streamlit run home.py
# Página para demonstração de projetos de Machine Learning e Data Science
# Masiero, Pedro e Azzi - 06/2021
#
#########################################################################################################

import streamlit as st
import similaridade
import obj_detection

def main():
	st.set_page_config(
		page_title="DBLab | Machine Learning Playground",
		page_icon="🤖")
	st.image('https://dblab.io/assets/images/dblab-logo-240x69.png')
	

	menu_activities = [	"Home",
						"Visão Computacional", # Modificar/inserir
						"Linguagem Natural (NLP)",
						"Sobre"]

	choice = st.sidebar.selectbox("Selecione:", menu_activities)

	# Exemplo
	if choice == 'Título do App': # Modificar
		st.text('funcao.main()')# <--- função aqui!

	elif choice == 'Visão Computacional':
		menu_visao = [	"Detecção de Objetos",
						"Busca por Similaridade"]

		choice = st.sidebar.selectbox("Selecione:", menu_visao)

		if choice == 'Detecção de Objetos':
			obj_detection.main()

		elif choice == 'Busca por Similaridade':
			similaridade.main()

	elif choice == 'Sobre':
		st.subheader("Sobre o DBLab Machine Learning Playground 🤖")
		st.markdown("Construído com Streamlit.")
		st.markdown("Contato: [dblab@dbserver.com.br](mailto:dblab@dbserver.com.br)")
		st.markdown("Site: [dblab.io](https://dblab.io/)")

	else: # Home
		st.header("Machine Learning Playground 🤖")
		st.image('https://miro.medium.com/proxy/1*RW4D-aNsQVY9FK_caWF0BQ.jpeg')
		st.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")

		st.image(image=['https://user-images.githubusercontent.com/32513366/71764203-797da800-2ec3-11ea-9eb9-8bdca4f45152.jpg', 
						'https://blog.betrybe.com/wp-content/uploads/2020/09/colab.jpeg', 
						'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKVKHMOCrU_cE0UxhqVVbky134Rh4ixU9Yw7Z3IdtFv04HjQpSRZBqiZhv0ig-awoiQG8&usqp=CAU', 
						'https://repository-images.githubusercontent.com/83878269/a5c64400-8fdd-11ea-9851-ec57bc168db5'], 
				width=130)

if __name__ == "__main__":
    main()



