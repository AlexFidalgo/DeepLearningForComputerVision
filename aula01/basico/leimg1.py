import cv2
nome="lenna.jpg"
a=cv2.imread(nome,1)
cv2.imshow("janela",a)
cv2.waitKey()