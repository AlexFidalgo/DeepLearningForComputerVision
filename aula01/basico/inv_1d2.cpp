//inv_1d2.cpp: Usando indice unidimensional
#include <cekeikon.h>
int main()
{ Mat_<GRY> a;
  le(a,"mickey_reduz.bmp");
  for (unsigned i=0; i<a.total(); i++)
    if (a(i)==0) a(i)=255;
    else a(i)=0;
  imp(a,"inv_1d.bmp");
}