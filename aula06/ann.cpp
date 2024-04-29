#include "../procimagem.h"
using namespace ml;

int main() {
  MNIST mnist(14,true,true);
  mnist.le("/home/alex/cekeikon5/tiny_dnn/data");

  Mat_<float> ay2(mnist.ay.rows, 10); ay2.setTo(0.0);
  for (int i=0; i<mnist.ay.rows; i++) {
    int j=round(mnist.ay(i)); assert(0<=j && j<10);
    ay2(i,j)=+1.0;
  }

  double t1=tempo(); Ptr<ANN_MLP> mlp=ANN_MLP::create();
  Mat_<int> layersSize = (Mat_<int>(1,4) << mnist.ax.cols, 100, 30, 10); //196
  mlp->setLayerSizes(layersSize);
  mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);
  TermCriteria termCrit = TermCriteria(TermCriteria::Type::MAX_ITER, 100, 0.001);
  mlp->setTermCriteria(termCrit);
  mlp->train(mnist.ax, ROW_SAMPLE, ay2);

  double t2=tempo();
  Mat_<float> saida; //(1,mnist.qx.cols);
  for (int l=0; l<mnist.nq; l++) {
    mlp->predict(mnist.qx.row(l),saida);
    mnist.qp(l)=argMax(saida);
  }
  double t3=tempo();

  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}