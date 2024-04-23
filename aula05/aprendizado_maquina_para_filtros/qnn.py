import cv2
import os
import time
import numpy as np

def NearestNeighbors(AX, AY, QX):
    QP = np.ones_like(QX) * 255

    for l in range(1, QX.shape[0] - 1):
        if l % 10 == 0:
            print(l)
        for c in range(1, QX.shape[1] - 1):
            qx = vizinhanca33(QX, l, c)
            mindist = np.iinfo(int).max
            minsai = 0
            for l2 in range(1, AX.shape[0] - 1):
                for c2 in range(1, AX.shape[1] - 1):
                    ax = vizinhanca33(AX, l2, c2)
                    dist = hamming(qx, ax)
                    if dist < mindist:
                        mindist = dist
                        minsai = AY[l2, c2]
                    if mindist == 0:
                        break
            QP[l, c] = minsai

    return QP

def vizinhanca33(a, lc, cc):
    d = np.zeros((1, 9), dtype=np.uint8)
    i = 0
    for l in range(-1, 2):
        for c in range(-1, 2):
            d[0, i] = a[lc + l, cc + c]
            i += 1
    return d

def hamming(a, b):
    if a.size != b.size:
        raise ValueError("Erro hamming")
    soma = np.sum(a != b)
    return soma

if __name__ == "__main__":
    current_directory = os.path.abspath(os.path.dirname(__file__))
    AX = cv2.imread(os.path.join(current_directory, "ax.bmp"), 0)
    AY = cv2.imread(os.path.join(current_directory, "ay.bmp"), 0)
    QX = cv2.imread(os.path.join(current_directory, "qx.bmp"), 0)
    
    start_time = time.time()
    result = NearestNeighbors(AX, AY, QX)
    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))
    
    cv2.imwrite(os.path.join(current_directory, "x.pgm"), result)
