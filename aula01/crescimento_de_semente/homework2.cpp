/*
Escreva um programa que lê uma imagem binária e imprime o número de componentes 
conexos da imagem. Rodando o seu programa para imagem “letras.bmp”, deve retornar 31. 
Aqui, estou chamando de componente conexo o conjunto de pixels pretos grudados entre si. 
Execute o programa também para imagens c2.bmp e c3.bmp.
*/

#include <opencv2/opencv.hpp>
#include <queue>
#include <cmath>
using namespace std;
using namespace cv;

int distancia(Vec3b a, Vec3b b) {
  return sqrt( pow(a[0]-b[0],2) + pow(a[1]-b[1],2) + pow(a[2]-b[2],2) );
}

Mat_<Vec3b> pintaAzul(Mat_<Vec3b> a, int ls, int cs) {

  Mat_<Vec3b> b=a.clone();
  queue<int> q;
  Vec3b preto(0,0,0);

  q.push(ls); q.push(cs);
  while (!q.empty()) {
    int l=q.front(); q.pop();
    int c=q.front(); q.pop();
    if (distancia(preto, b(l,c)) < 50) {
      b(l,c)=Vec3b(255,0,0);
      q.push(l-1); q.push(c); 
      q.push(l+1); q.push(c);
      q.push(l); q.push(c+1);
      q.push(l); q.push(c-1); 
    }
  }
  return b;
}

int main() {

    Mat_<Vec3b> a;
    Mat_<Vec3b> b;
    int l, c, n_comp;

    a = imread("letras.bmp", 1);
    b = a.clone(); 
    Vec3b preto(0,0,0);
    n_comp = 0;

    for (l = 0; l < a.rows - 1; l++) {
        for (c = 0; c < a.cols - 1; c++) {

            if (distancia(preto, a(l,c)) < 50 && b(l,c) != Vec3b(255,0,0)) {
                n_comp += 1;
                b = pintaAzul(b, l, c);
            }

        }
    }

    cout << "Numero de componentes = " << n_comp << endl;

}

