// rotacaocv.cpp

#include "procimagem.h"

// Main function
int main() { 
    // Reading input image (grayscale)
    Mat_<uchar> ent = imread("lennag.jpg", 0);

    // Declaring output image
    Mat_<uchar> sai;

    // Creating a rotation matrix for a rotation around the center of the input image
    Mat_<double> m = getRotationMatrix2D(Point2f(ent.cols / 2, ent.rows / 2), 270, 1);

    // Displaying the rotation matrix
    cout << m << endl;

    // Applying the rotation transformation to the input image
    warpAffine(ent, sai, m, ent.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));

    // Writing the rotated image to an output file named "rotacao_cv.jpg"
    imwrite("rotacao_cv.jpg", sai);
}
