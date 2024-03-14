//inv_ocv2.cpp feito em OpenCV puro (sem usar template)
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main() { 
  Mat a;
  a=imread("mickey_reduz.bmp",0);
  for (int l=0; l<a.rows; l++)
    for (int c=0; c<a.cols; c++)
      if (a.at<uchar>(l,c)==0) a.at<uchar>(l,c)=255;
      else a.at<uchar>(l,c)=0;
  imwrite("inv_ocv.pgm",a);
}