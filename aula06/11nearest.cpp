#include "../procimagem.h"
#include <iostream>
#include <unordered_map>

float get_mode(cv::Mat_<float> v) {
    std::unordered_map<float, int> found_values;

    for (int i = 0; i < v.rows; ++i) {
        for (int j = 0; j < v.cols; ++j) {
            float elem = v(i, j);
            found_values[elem] = found_values[elem] + 1;
        }
    }

    float mode = 0;
    int max_count = 0;
    for (auto& pair : found_values) {
        if (pair.second > max_count) {
            mode = pair.first;
            max_count = pair.second;
        }
    }

    return mode;
}

int main() {
  MNIST mnist(14, true, true);
  mnist.le("/home/alex/cekeikon5/tiny_dnn/data");
  double t1=tempo();
  Ptr<ml::KNearest>  knn(ml::KNearest::create());
  knn->train(mnist.ax, ml::ROW_SAMPLE, mnist.ay);
  double t2=tempo();

  // Precisamos classificar 10000 imagens de teste mnist.qx (10000x196)
  // Precisamos colocar as 10000 classificacoes em mnist.qp (10000x1)
  Mat_<float> saidas,dists;
  int k=3;
  for (int l=0; l<10000; l++) {
    knn->findNearest(mnist.qx.row(l), k, noArray(), saidas, dists);
    //cout << saidas.rows << " " << saidas.cols << endl;  // isso imprime 1 k
    //cout << saidas << endl;                             // isso imprime uma matriz de 1 linha e k colunas com os valores das k imagens mais próximas obtidas
    //cout << dists.rows << " " << dists.cols << endl;    // isso imprime 1 k
    //cout << dists << endl;                              // isso imprime uma matriz de 1 linha e k colunas com as distâncias entre as imagens obtidas e a imagem em qx.row(l)
    // Calcule a moda (o elemento mais frequente) de saidas.
    // Coloque moda em mnist.qp(l)
    mnist.qp(l)=get_mode(saidas);
  }

  double t3=tempo();
  printf("Erros=%10.2f%%\n",100.0*mnist.contaErros()/mnist.nq);
  printf("Tempo de treino: %f\n",t2-t1);
  printf("Tempo de predicao: %f\n",t3-t2);
}