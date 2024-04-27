#include "../procimagem.h"

int main() {
  MNIST mnist(14, true, true);
  mnist.le("/home/alex/cekeikon5/tiny_dnn/data");
  double t1=tempo();
  flann::Index ind(mnist.ax,flann::KDTreeIndexParams(4)); //usando os dados de treino ax, crie 4 kd árvores
  double t2=tempo();
  Mat_<int> matches(mnist.na,1); Mat_<float> dists(mnist.na,1);
  
  for (int l=0; l<mnist.qx.rows; l++) {
    mnist.qp(l)=mnist.ay(matches(l));
  }
  double t3=tempo();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}

