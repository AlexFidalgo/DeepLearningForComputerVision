/*
Escreva um programa que elimina o ruído branco nas regiões 
pretas da imagem “mickey.bmp”. Crie "mickey_reduz_hw.bmp"
*/

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {

    Mat_<uchar> a;
    a = imread("mickey.bmp", 0);
    Mat_<uchar> b = a.clone();
    int l, c;
    for (l = 1; l < a.rows - 1; l++) {
        for (c = 1; c < a.cols - 1; c++) {
            if (a(l,c) != a(l+1,c) &&
                a(l,c) != a(l-1,c) &&
                a(l,c) != a(l,c+1) &&
                a(l,c) != a(l,c-1)) {
                    b(l,c) = a(l+1,c);
                }
        }
    }
    imwrite("mickey_reduz_hw.bmp", b);
}

