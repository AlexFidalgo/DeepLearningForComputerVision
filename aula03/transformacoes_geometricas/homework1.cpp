/*
Corrija a deformação em perspectiva do tabuleiro de xadrez abaixo, gerando uma imagem onde 
cada casa do tabuleiro é um quadrado alinhado aos eixos do sistema de coordenadas. 
Consequentemente, o tabuleiro todo será um retângulo alinhado aos eixos do sistema de 
coordenadas.
Nota: Você não precisa determinar automaticamente as esquinas do tabuleiro. Pode colocar no 
seu programa, manualmente, as suas coordenadas. 
*/

#include "procimagem.h"

int main() {
  Mat_<float> src = (Mat_<float>(4,2) <<
    139,45,
    108,294,
    322,35,
    352,293);
  Mat_<float> dst = (Mat_<float>(4,2) <<
    109,33,
    109,295,
    323,35,
    323,293);
  Mat_<double> m=getPerspectiveTransform(src,dst);
  cout << m << endl;

  //Verifica se a transformacao esta fazendo o que queremos
  Mat_<double> v=(Mat_<double>(3,1) << -22,479,1);
  Mat_<double> w=m*v;
  cout << w << endl;
  cout << w(0)/w(2) << " " << w(1)/w(2) << endl;

  //Corrige a perspectiva
  Mat_<Vec3b> a; a = imread("calib_result.jpg", 1);
  Mat_<Vec3b> b;
  warpPerspective(a,b,m,a.size());
  imwrite("homework_result.jpg", b);

}
