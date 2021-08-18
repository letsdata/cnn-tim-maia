# Importar as bibliotecas
# Precisamos do streamlit, numpu, PIL e o pytorch
# Aqui precisamos só carregar o modelo já treinado!
import streamlit as st
import numpy as np
from PIL import Image
import time

import torch
from torchvision import transforms

# definindo as transformações para imagens novas a serem submetidas ao modelo!
image_size = 100

# Transformando as imagens
redimensionamento_imagem = transforms.Compose([
        transforms.Resize(size=[image_size, image_size]),
        transforms.ToTensor(),
    ])

def predicao_tim_maia(model, test_image):
    '''
    Função para realizar a predição do status do AR
    Parâmetros
        :param model: modelo para testar
        :param test_image_name: imagem teste
    '''
    transform = redimensionamento_imagem

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size)

    # Não precisa atualizar os coeficientes do modelo
    with torch.no_grad():
        model.eval()

        # Modelo retorna as probabilidades em log (log softmax)
        out = model(test_image_tensor)

        # torch.exp para voltar a probabilidade de log para a probabilidade linear
        ps = torch.exp(out)

        # topk retorna o os k maiores valores do tensor
        # o tensor de probabilidades vai trazer na 1a posição a classe com maior
        # probabilidade de predição
        topk, topclass = ps.topk(3, dim=1)



        classe_com_maior_prob = np.argmax(topk.cpu().numpy()[0])

    return topclass[0][0]


# Designing the interface
st.title("Tim Maia - Chocolate com Pytorch")
# For newline
st.write('\n')

image = Image.open('images/tim_maia.png')
show = st.image(image, use_column_width=True)

st.sidebar.title("Suba uma imagem de bebida pro nosso querido Tim Maia!")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['jpg', 'jpeg'])

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Imagem enviada', use_column_width=True)
    # We preprocess the image to fit in algorithm.

# For newline
st.sidebar.write('\n')

# Carregar o modelo

modelo = torch.load('./modelos/melhor_modelo.pt')

if st.sidebar.button("Clique aqui para saber se o Tim Maia vai gostar da bebida ou não!"):
    if uploaded_file is None:

        st.sidebar.write("Suba uma imagem para o Tim Maia")

    else:

        with st.spinner('Tim Maia experimentando...'):

            prediction = predicao_tim_maia(modelo, u_img)
            time.sleep(2)
            st.success('Pronto!')

        st.sidebar.header("Tim Maia disse que a imagem que você subiu é de...")


        print(prediction)

        if prediction == 0:
            st.sidebar.write("É chocolate!!! Me liguei no chocolate!!!", '\n')
            show.image('./images/tim_maia_feliz.png', 'Tim Maia tá felizão!', use_column_width=True)
        elif prediction == 1:
            st.sidebar.write("Não quero coca-cola, me liguei no chocolate! Só quero chocolate!", '\n')
            show.image('./images/tim_maia_pistola.png', 'Tim Maia tá pistola!', use_column_width=True)
        elif prediction == 2:
            st.sidebar.write("Não adianta vir com guaraná pra mim. É chocolate o que eu quero beber!!", '\n')
            show.image('./images/tim_maia_pistola.png', 'Tim Maia tá pistola!', use_column_width=True)
