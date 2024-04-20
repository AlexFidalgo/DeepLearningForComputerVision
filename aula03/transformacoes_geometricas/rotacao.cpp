#include "procimagem.h"

// Function to convert degrees to radians
inline double deg2rad(double x) {
    return (x / 180.0) * (M_PI);
}

int main(int argc, char** argv) {
    
    if (argc != 4) {
        printf("rotacao ent.pgm sai.pgm graus\n");
        erro("Erro: Numero de argumentos invalido");
    }

    double graus;
    // Extracting the rotation angle from command line arguments
    sscanf(argv[3], "%lf", &graus);
    // Converting degrees to radians
    double radianos = deg2rad(graus);
    // Calculating cosine and sine of the angle
    double co = cos(radianos);
    double se = sin(radianos);

    // Reading input image
    ImgXyb<uchar> a = imread(argv[1], 0);
    // Setting center and background color of the input image
    a.centro(a.rows / 2, a.cols / 2);
    a.backg = 255;

    // Creating output image
    ImgXyb<uchar> b(a.rows, a.cols);
    // Setting center and background color of the output image
    b.centro(b.rows / 2, b.cols / 2);
    b.backg = 255;

    // Iterating through each pixel of the output image
    for (int xb = b.minx; xb <= b.maxx; xb++) {
        for (int yb = b.miny; yb <= b.maxy; yb++) {
            // Applying rotation transformation
            int xa = cvRound(xb * co + yb * se);
            int ya = cvRound(-xb * se + yb * co);
            // Assigning pixel value from input image to output image
            b(xb, yb) = a(xa, ya);
        }
    }

    // Writing the rotated image to output file
    imwrite(argv[2], b);
}
