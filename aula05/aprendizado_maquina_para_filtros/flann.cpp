#include "procimagem.h"

int main() {
  Mat_<Vec3b> ax=imread("ax.png",1);
  Mat_<uchar> ay=imread("ay.png",0);
  Mat_<Vec3b> qx=imread("f1.jpg",1);
  if (ax.size()!=ay.size()) erro("Erro dimensao");
  Mat_<uchar> qp(qx.rows,qx.cols);

  //Cria as estruturas de dados para alimentar OpenCV (conversão das imagens AX, AY para o formato desejado por OpenCV)
  Mat_<float> features(ax.rows*ax.cols,3);
  Mat_<int> saidas(ax.rows*ax.cols,1);
  int i=0;
  for (int l=0; l<ax.rows; l++)
    for (int c=0; c<ax.cols; c++) { 
      features(i,0)=ax(l,c)[0]/255.0; // Matriz features tem 3 colunas, para armazenar as cores BGR
      features(i,1)=ax(l,c)[1]/255.0; // convenção de que cores BGR em uint8 (0 a 255) são convertidas para intervalo [0,1] quando forem armazenadas como float.
      features(i,2)=ax(l,c)[2]/255.0;
      saidas(i)=ay(l,c);
      i=i+1;
    }
  flann::Index ind(features,flann::KDTreeIndexParams(4)); // Cria 4 kd-árvores para buscar o vizinho próximo

  // Aqui começa a busca pelo vizinho mais próximo    
  Mat_<float> query(1,3);
  vector<int> indices(1);
  vector<float> dists(1);
  for (int l=0; l<qp.rows; l++)
    for (int c=0; c<qp.cols; c++) {
      query(0)=qx(l,c)[0]/255.0;
      query(1)=qx(l,c)[1]/255.0;
      query(2)=qx(l,c)[2]/255.0;
      ind.knnSearch(query,indices,dists,1,flann::SearchParams(0)); // Buscas feitas nas 4 árvores e é escolhido o exemplo mais parecido com a 
      // instância de teste qx. Fornecemos o query e ele retorna indices dos vizinhos mais próximos e as distancias para esses vizinhos mais proximos.
      // 1 indica que retornará apenas 1 vizinho mais próximo. O 0 no quinto parametro indica que não fará backtracking
      qp(l,c)=saidas(indices[0]);
    }
  imwrite("f1-flann.png",qp);
}