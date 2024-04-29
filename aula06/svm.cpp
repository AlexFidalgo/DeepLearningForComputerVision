
#include "../procimagem.h"

using namespace ml;

int main() {
  MNIST mnist(10, true, false); //10x10 e sem Bounding Box
  mnist.le("/home/hae/cekeikon5/tiny_dnn/data"); 
  Ptr<SVM> ind=SVM::create();
  double t1=tempo();
  ind->train(mnist.ax, ROW_SAMPLE, mnist.AY); // Treinamento da SVM
  double t2=tempo();
  for (int l=0; l<mnist.nq; l++) 
    mnist.qp(l)=ind->predict(mnist.qx.row(l));
  double t3=tempo();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}
