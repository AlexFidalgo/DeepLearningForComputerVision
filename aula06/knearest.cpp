#include "../procimagem.h"

int main() {
  MNIST mnist(14, true, true);
  mnist.le("/home/alex/cekeikon5/tiny_dnn/data");
  double t1=tempo();
  Ptr<ml::KNearest>  knn(ml::KNearest::create());
  knn->train(mnist.ax, ml::ROW_SAMPLE, mnist.ay);
  double t2=tempo();
  Mat_<float> dist;
  knn->findNearest(mnist.qx, 1, noArray(), mnist.qp, dist);
  double t3=tempo();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}