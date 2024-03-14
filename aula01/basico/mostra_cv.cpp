#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {
    Mat a=imread("lenna.jpg",1);
    imshow("janela",a);
    waitKey(0);
}