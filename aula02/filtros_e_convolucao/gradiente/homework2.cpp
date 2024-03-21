/*
Complete o programa, fazendo operações de módulo pixel a pixel (abs), inverter sinal 
pap (- com um argumento), soma pap (+), subtração pap (- com dois argumentos), mínimo 
pap (min), máximo pap (max) e elevar/raiz (pow) com as matrizes sx e sy, para obter as 
imagens com:
    • Todas as bordas verticais: ver_todo.png
    • A borda vertical esquerda: ver_esq.png
    • A borda vertical direita: ver_dir.png.
    • Todas bordas horizontais: hor_todo.png 
    • A borda horizontal superior: hor_sup.png
    • A borda horizontal inferior: hor_inf.png
    • Módulo do gradiente: modulo.png
*/

#include "procimagem.h"
#include <cmath>

int main() {
  Mat_<float> a = imread("circulo.png", 0);
  Mat_<float> sx, sy, ox, oy;
  Sobel(a, sx, -1, 1, 0, 3); 
  ox = sx / 4.0 + 128; 
  imwrite("ox.png", ox);
  Sobel(a, sy, -1, 0, 1, 3); 
  oy = sy / 4.0 + 128; 
  imwrite("oy.png", oy);

  // Todas as bordas verticais
  Mat_<float> ver_todo = abs(sx);
  imwrite("ver_todo.png", ver_todo);

  // Borda vertical esquerda
//   Mat_<float> ver_esq = sx;
//   ver_esq(ver_esq < 0) = 0;
//   imwrite("ver_esq.png", ver_esq);

//   // Borda vertical direita
//   Mat_<float> ver_dir = -sx;
//   ver_dir(ver_dir < 0) = 0;
//   imwrite("ver_dir.png", ver_dir);

//   // Todas as bordas horizontais
//   Mat_<float> hor_todo = abs(sy);
//   imwrite("hor_todo.png", hor_todo);

//   // Borda horizontal superior
//   Mat_<float> hor_sup = sy;
//   hor_sup(hor_sup < 0) = 0;
//   imwrite("hor_sup.png", hor_sup);

//   // Borda horizontal inferior
//   Mat_<float> hor_inf = -sy;
//   hor_inf(hor_inf < 0) = 0;
//   imwrite("hor_inf.png", hor_inf);

//   // Módulo do gradiente
//   Mat_<float> modulo = sqrt(pow(sx, 2) + pow(sy, 2));
//   imwrite("modulo.png", modulo);

  return 0;
}
