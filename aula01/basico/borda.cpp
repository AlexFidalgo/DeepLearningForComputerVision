#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {
    Mat_<uchar> a = imread("mickey_reduz.bmp", 0);
    Mat_<uchar> b(a.rows, a.cols, 255);
    for (int l = 1; l < a.rows - 1; l++) {
        for (int c = 1; c < a.cols - 1; c++) {
            if (a(l,c) == 0 && (
                a(l-1,c) != 0 || a(l+1,c) != 0 ||
                a(l,c-1) != 0 || a(l,c+1) != 0)
            ) {
                b(l,c) = 0;
            } else {
                b(l,c) = 255;
            }
        }
    }
    imwrite("borda.bmp", b);
}
