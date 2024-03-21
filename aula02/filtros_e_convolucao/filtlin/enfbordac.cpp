//enfbordac.cpp - 2024
#include "procimagem.h"

int main(int argc, char** argv) {
  if (argc!=3) erro("enfborda ent.ppm sai.ppm");

  Mat_<Vec3f> ent=imread(argv[1],1);
  if (ent.empty())
    erro("Error loading input image.");

  Mat_<float> ker= (Mat_<float>(3,3) <<
                -1, -1, -1,
                -1, 18, -1,
                -1, -1, -1);
  ker = (1.0/10.0) * ker;
  Mat_<Vec3f> sai = filtro2d(ent,ker);
  imwrite(argv[2],sai);
}