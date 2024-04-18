// Especifica numero de linhas e numero de colunas
#include "procimagem.h"

int main() {

    Mat_<uchar> a = imread("lennag.jpg", 0);

    int nl = 740;
    int nc = 625;

    Mat_<uchar> b(nl, nc);

    for (int lb=0; lb < b.rows; lb++) {

        for (int cb=0; cb < b.cols; cb++) {

            int la = round(double(lb*a.rows)/b.rows);
            int ca = round(double(cb*a.cols)/b.cols);
            b(lb,cb) = a(la, ca);
        }

    }

    imwrite("vizinho.png", b);


}