#include "procimagem.h"

int main() {
    // Read image 'bbox.pgm' and normalize
    Mat_<float> a = imread("bbox.pgm", 0);
    a = a / 255.0;

    // Read image 'letramore.pgm', normalize, perform somaAbsDois and dcReject operations
    Mat_<float> q = imread("letramore.pgm", 0);
    q = q / 255.0;
    q = somaAbsDois(dcReject(q));

    // Perform filtro2d operation on image 'a' using modified template 'q'
    Mat_<float> p = filtro2d(a, q);

    // Write correlation result to 'correlacao.png'
    imwrite("correlacao.png", 255 * p);

    // Convert image 'a' to color (BGR) format
    Mat_<Vec3f> d;
    cvtColor(a, d, COLOR_GRAY2BGR);

    // Draw rectangles on image 'd' based on correlation result 'p'
    for (int l = 0; l < a.rows; l++) {
        for (int c = 0; c < a.cols; c++) {
            if (p(l, c) >= 0.999) {
                // Draw rectangle around detected object
                rectangle(d, Point(c - 109, l - 38), Point(c + 109, l + 38), Scalar(0, 0, 1), 3);
            }
        }
    }

    // Write image with rectangles to 'ocorrencia.png'
    imwrite("ocorrencia.png", 255 * d);

    return 0;
}
