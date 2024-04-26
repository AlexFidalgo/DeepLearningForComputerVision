//visualiza.cpp - imprime as primeiras 9 imagens de teste de MNIST 2024
#include "../procimagem.h"


int main() {
  MNIST mnist(28,true,false); //nao redimensiona imagens (por isso o 28, se fosse 14 iria diminuir pra 14)
                              //inverte preto/branco=true,
                              //crop bounding box=false
  mnist.le("/home/alex/cekeikon5/tiny_dnn/data");
  mnist.qp.setTo(2); //Coloca uma classificacao errada de proposito
  Mat_<uchar> e=mnist.geraSaidaErros(3,3); //Organiza 10000 imagens em 100 linhas e 100 colunas
  imwrite("visualiza.png",e); //Imprime 10000 imagens-teste como visual.png
}