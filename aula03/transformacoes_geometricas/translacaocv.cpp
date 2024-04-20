//translacaocv.cpp

#include "procimagem.h"

int main() {

    Mat_<uchar> ent = imread("lennag.jpg", 0);
    Mat_<uchar> sai;

    Mat_<double> m = (Mat_<double> (2,3) << 1, 0, 50,
                                            0, 1, 25);
    cout << m << endl;

    warpAffine(ent, sai, m, ent.size(), INTER_LINEAR, BORDER_REFLECT, Scalar(255));

    imwrite("translacao.jpg", sai);
}