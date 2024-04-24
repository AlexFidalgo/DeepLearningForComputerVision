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


if __name__ == "__main__":

    current_directory = os.path.abspath(os.path.dirname(__file__))

    # AX = cv2.imread(os.path.join(current_directory, "janei.pgm"), 0)
    # AY = cv2.imread(os.path.join(current_directory, "janei-1.pgm"), 0)
    # QX = cv2.imread(os.path.join(current_directory, "julho.pgm"), 0)
    # window_size = 3
    
    res_julho = os.path.join(current_directory, "julho-p1.pgm")
    # result = NearestNeighbors(AX, AY, QX, window_size)    
    # cv2.imwrite(res_julho), result)

    res_julho_filtrada = cv2.imread(res_julho, 0)
    for _ in tqdm(range(5)):
        res_julho_filtrada = median_filter(res_julho_filtrada)

    cv2.imwrite(os.path.join(current_dir, "res_julho_filtrada.png"), res_julho_filtrada)
