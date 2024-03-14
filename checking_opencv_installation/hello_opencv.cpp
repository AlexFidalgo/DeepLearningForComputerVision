#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() { 
    Mat_<Vec3b> a(300,300, Vec3b(255,255,0));
    for (int l=100; l<200; l++)
        for (int c=100; c<200; c++)
            a(l,c)=Vec3b(0,255,255);
    namedWindow("janela");
    imshow("janela",a);
    waitKey();
}