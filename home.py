#########################################################################################################
#
# DBLab
# Machine Learning Playground
# 
# >> streamlit run playground.py
# Página para demonstração de projetos de Machine Learning e Data Science
# Masiero, Pedro e Azzi - 06/2021
#
#########################################################################################################

import streamlit as st
import similaridade #importando o script 'similaridade.py' com minha feature
import obj_detection

def main():
	st.set_page_config(
		page_title="DBLab | Machine Learning Playground",
		page_icon="🤖")
	st.image('https://dblab.io/assets/images/dblab-logo-240x69.png')
	

	activities = [	"Home",
					"Detecção de Objetos", # Modificar/inserir
					"Busca por Similaridade",
					"Sobre"]

	choice = st.sidebar.selectbox("Selecione:",activities)

	# Exemplo
	if choice == 'Busca por Similaridade':
		similaridade.main() # rodando minha feature chamando a main

	elif choice == 'Título do App': # Modificar
		st.text('funcao.main()')# <--- função aqui!

	elif choice == 'Detecção de Objetos':
		obj_detection.main()

	elif choice == 'Sobre':
		st.subheader("Sobre o DBLab Machine Learning Playground 🤖")
		st.markdown("Construído com Streamlit by [JCharisTech](https://www.jcharistech.com/); Catálogo by Netshoes.")
		st.markdown("Contato: [dblab@dbserver.com.br](mailto:dblab@dbserver.com.br)")
		st.markdown("Site: [dblab.io](https://dblab.io/)")

	else: # Home
		st.header("Machine Learning Playground 🤖")
		st.image('https://miro.medium.com/proxy/1*RW4D-aNsQVY9FK_caWF0BQ.jpeg')
		st.markdown("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")

if __name__ == "__main__":
    main()



