/*
Cada uma das 12 imagens q??.jpg aparece uma única vez na imagem a.jpg, possivelmente 
rotacionado. Faça um programa que lê as imagens a.jpg e as 12 imagens-modelos q??.jpg e 
gera a imagem p.jpg indicando onde está cada uma das 12 imagens-modelos juntamente com o 
ângulo da rotação, como na figura abaixo à direita.

Sugestão: Você pode rotacionar as imagens q??.jpg em vários ângulos e buscá-los todos na 
imagem a.jpg.
*/

#include "procimagem.h"

Mat_<uchar> rotacao(Mat_<uchar> ent, double graus, Point2f centro, Size tamanho) {
  Mat_<double> m=getRotationMatrix2D(centro, graus, 1.0);
  Mat_<uchar> sai;
  warpAffine(ent, sai, m, tamanho, INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
  return sai;
}

int main() {

    const float borderWidthPercentage = 0.00; // 5% of the image's width
    const float borderColor = 180.0 / 255.0; // Specific color

    Mat_<float> inputImage = imread("a1.jpg", 0);
    inputImage = inputImage / 255;

    Mat_<Vec3f> d;
    cvtColor(inputImage, d, COLOR_GRAY2BGR);

    for (int i = 0; i <= 12; ++i) {

        string filename = "q" + to_string(i / 10) + to_string(i % 10) + ".jpg";
        cout << "Attempting to load: " << filename << endl;
        Mat_<float> templateImage = imread(filename, 0);
        templateImage = templateImage / 255;

        const int borderWidth = static_cast<int>(templateImage.cols * borderWidthPercentage);
        for (int r = 0; r < templateImage.rows; ++r) {
            for (int c = 0; c < templateImage.cols; ++c) {
                // Check if the pixel is within the border region
                if (r < borderWidth || r >= templateImage.rows - borderWidth ||
                    c < borderWidth || c >= templateImage.cols - borderWidth) {
                    templateImage(r, c) = borderColor;
                }
            }
        }

        Mat_<float> borderedTemplateImage = dcReject(templateImage, borderColor);

        for (double angle = 0.0; angle < 360.0; angle += 5) {
            
            Mat_<float> rotatedTemplateImage = rotacao(borderedTemplateImage, angle, 
                                                       Point2f(templateImage.size())/2, templateImage.size());
            
            Mat_<float> correlation = matchTemplateSame(inputImage, rotatedTemplateImage, TM_CCOEFF_NORMED);

            
            for (int l = 0; l < inputImage.rows; l++) {
                for (int c = 0; c < inputImage.cols; c++) {
                    if (correlation(l, c) >= 0.9999) {
                        rectangle(d, Point(c - 1, l - 1), Point(c + 1, l + 1), Scalar(0, 0, 1), 3);
                    }
                }
            }
            
        }
    }

    imwrite("resultado_homework2.png",255*d);

    return 0;
}
