# Resolva o problema de manter as letras “a” e “A” e apaga as outras letras da imagem usando 
# algum método de aprendizado de máquina (pode escolher o método livremente). As sugestões são 
# vizinho mais próximo “força bruta”, vizinho mais próximo aproximado usando kd-árvore ou árvore 
# de decisão. Pode se usar como exemplos os programas que vão aparecer nesta apostila mais 
# adiante. O seu programa só precisa funcionar para o tipo de fonte das imagens fornecidas 
# (lax.bmp, lay.bmp e lqx.bmp).

import numpy as np
from sklearn.neighbors import KDTree
import cv2
import os
import time
from tqdm import tqdm
import numpy as np

def reverse_transform_index(index, A):

    l, c = A.shape
    l_index = index // c
    c_index = index % c
    return l_index, c_index


def get_vizinhanca_matrix(A, window_size):
    l, c = A.shape
    padding = window_size // 2
    window_shape = (1, window_size * window_size)

    C = np.zeros((l * c, window_shape[1]), dtype=np.uint8)

    for i in range(l):
        for j in range(c):
            index = i * c + j
            if i < padding or i >= l - padding or j < padding or j >= c - padding:
                C[index] = np.full(window_shape, 255, dtype=np.uint8)
            else:
                C[index] = get_vizinhanca(A, i, j, window_size)

    return C

def get_vizinhanca(a, lc, cc, window_size):
    d = np.zeros((1, window_size * window_size), dtype=np.uint8)
    index = 0
    padding = window_size // 2

    for l in range(-padding, padding + 1):
        for c in range(-padding, padding + 1):
            d[0, index] = a[lc + l, cc + c]
            index += 1

    return d

def NearestNeighbors(AX, AY, QX, window_size):

    viz_AX = get_vizinhanca_matrix(AX, window_size)
    viz_QX = get_vizinhanca_matrix(QX, window_size)

    QP = []

    tree = KDTree(viz_AX)

    for row in viz_QX:

        ind = tree.query(row.reshape(1,-1))[1]
        index = reverse_transform_index(ind[0,0], AX)
        QP.append(AY[index])

    QP = np.array(QP).reshape((QX.shape[0], QX.shape[1]))

    return QP

def median_filter(img):
    filtered_img = cv2.medianBlur(img, 3)
    return filtered_img

def make_black_pixels_red(gray_img):
    colored_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)
    colored_img[gray_img == 0] = [0, 0, 255]  # [B, G, R]
    colored_img[~gray_img==0] = [255, 255, 255]
    return colored_img

import cv2

def sobrepoe_imagens(QX, jul_red):
    """
    Args:
        QX: Grayscale image (cv2 format).
        jul_red: Colored image (cv2 format).
    """

    QX_color = cv2.cvtColor(QX, cv2.COLOR_GRAY2BGR) # converte imagem grayscale para colorida

    lower_red = np.array([0, 0, 200])
    upper_red = np.array([100, 100, 255])

    mask = cv2.inRange(jul_red, lower_red, upper_red) # cria uma mask selecionando pixels vermelhos em jul_red

    result = cv2.bitwise_and(jul_red, jul_red, mask=mask) # aplica bitwise_and pra preservar pixels vermelhos

    inverted_mask = cv2.bitwise_not(mask) # seleciona areas nao vermelhas

    grayscale_result = cv2.bitwise_and(QX_color, QX_color, mask=inverted_mask) # aplica bitwise_and pra preservar grayscale

    final_image = cv2.add(result, grayscale_result) #combina areas vermelhas e grayscale

    return final_image



if __name__ == "__main__":

    current_directory = os.path.abspath(os.path.dirname(__file__))

    # AX = cv2.imread(os.path.join(current_directory, "janei.pgm"), 0)
    # AY = cv2.imread(os.path.join(current_directory, "janei-1.pgm"), 0)
    QX = cv2.imread(os.path.join(current_directory, "julho.pgm"), 0)
    # window_size = 3
    
    res_julho = os.path.join(current_directory, "julho-p1.pgm")
    # result = NearestNeighbors(AX, AY, QX, window_size)    
    # cv2.imwrite(res_julho), result)

    res_julho_filtrada = cv2.imread(res_julho, 0)
    for _ in tqdm(range(5)):
        res_julho_filtrada = median_filter(res_julho_filtrada)

    jul_red = make_black_pixels_red(res_julho_filtrada)

    cv2.imwrite(os.path.join(current_directory, "jul_red.png"), jul_red)

    sobreposto = sobrepoe_imagens(QX, jul_red)
    cv2.imwrite(os.path.join(current_directory, "julho-c1.png.png"), sobreposto)
