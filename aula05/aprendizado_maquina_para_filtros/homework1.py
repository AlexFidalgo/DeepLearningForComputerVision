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
import numpy as np

def reverse_transform_index(index, A):

    l, c = A.shape
    l_index = index // c
    c_index = index % c
    return l_index, c_index


def get_vizinhanca_matrix(A):

    l, c = A.shape
    C = np.zeros((l * c, 49), dtype=np.uint8)

    for i in range(l):
        for j in range(c):
            index = i * c + j
            if i < 3 or i >= l - 3 or j < 3 or j >= c - 3:
                C[index] = np.full((1, 49), 255, dtype=np.uint8)
            else:
                C[index] = vizinhanca77(AX, i, j)

    return C

def vizinhanca33(a, lc, cc):
    d = np.zeros((1, 9), dtype=np.uint8)
    i = 0
    for l in range(-1, 2):
        for c in range(-1, 2):
            d[0, i] = a[lc + l, cc + c]
            i += 1
    return d

def vizinhanca77(a, lc, cc):
    d = np.zeros((1, 49), dtype=np.uint8)
    i = 0
    for l in range(-3, 4):
        for c in range(-3, 4):
            d[0, i] = a[lc + l, cc + c]
            i += 1
    return d


def NearestNeighbors(AX, AY, QX):

    viz_AX = get_vizinhanca_matrix(AX)
    viz_QX = get_vizinhanca_matrix(QX)

    QP = []

    tree = KDTree(viz_AX)

    for row in viz_QX:

        ind = tree.query(row.reshape(1,-1))[1]
        index = reverse_transform_index(ind[0,0], AX)
        QP.append(AY[index])

    QP = np.array(QP).reshape((QX.shape[0], QX.shape[1]))

    return QP


if __name__ == "__main__":
    current_directory = os.path.abspath(os.path.dirname(__file__))
    AX = cv2.imread(os.path.join(current_directory, "lax.bmp"), 0)
    AY = cv2.imread(os.path.join(current_directory, "lay.bmp"), 0)
    QX = cv2.imread(os.path.join(current_directory, "lqx.bmp"), 0)
    
    start_time = time.time()
    result = NearestNeighbors(AX, AY, QX)
    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))
    
    cv2.imwrite(os.path.join(current_directory, "lx_kd.pgm"), result)
