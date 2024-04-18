/*
Escreva um programa que detecta as 4 ocorrências de urso “q.png” na imagem a analisar “a.png” 
gerando uma imagem processada semelhante a “p.png”, chamando uma única vez a rotina de 
template matching. 

Nota: Este exercício pode ser resolvido usando template matching aplicado diretamente em 
níveis de cinza. Não detecte as bordas antes de chamar template matching. Detectar bordas é 
uma solução bem menos robusta do que fazer template matching direto em níveis de cinza.
Digamos que você tenha detectado as bordas das imagens a e q abaixo e tenha conseguido um 
casamento das bordas das duas imagens. Neste caso, as bordas de a e q deixam de “bater” 
se deslocar q em um pixel em qualquer direção, se mudar minimamente a escala ou fizer uma 
pequena rotação. Enquanto isso, o casamento levando em conta o nível de cinza das imagens 
continuam “batendo” mesmo com pequenas distorções. 
*/

#include "procimagem.h"

int main() {

    Mat_<float> inputImage = imread("a.png", 0);
    inputImage = inputImage/255;

    Mat_<float> templateImage = imread("q.png", 0);
    templateImage = templateImage/255;

    Mat_<float> correlation = matchTemplateSame(inputImage, templateImage, TM_CCOEFF_NORMED);

    Mat_<Vec3f> d;
    cvtColor(inputImage,d,COLOR_GRAY2BGR);
    for (int l=0; l<inputImage.rows; l++) {
        for (int c=0; c<inputImage.cols; c++) {
            if (correlation(l,c)>=0.9 || correlation(l,c) <= -0.9)
                rectangle(d,Point(c-3,l-3),Point(c+3,l+3),Scalar(0,0,255),FILLED);
            }   
    }
    imwrite("ocorrencia_homework1.png",255*d);

    return 0;
}