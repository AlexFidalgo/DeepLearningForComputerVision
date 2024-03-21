/*
Escreva um programa que usa o filtro mediana (usando a função medianBlur do OpenCV ou 
o filtro implementado “manualmente”) para filtrar a imagem ruidosa fever-1.pgm e f
ever-2.pgm (que se encontram dentro do arquivo “filtlin.zip” diretório “textura”) obtendo 
as imagens limpas. Figura 8 mostra a saída esperada filtrando fever-2.pgm.
*/

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {

  int i;
  
  Mat_<uchar> a=imread("fever-2.pgm",0);
  Mat_<uchar> b = a.clone();

  for (i = 0; i < 10; i++) {

    medianBlur(b, b, 5);

  }
  
  imwrite("saida_limpa_fever_2.pgm",b);
}
