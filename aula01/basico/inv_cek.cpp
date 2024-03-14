// inv_cek.cpp
#include <cekeikon.h>
int main()
    { Mat_<GRY> a;
    le(a,"mickey_reduz.bmp");
    for (int l=0; l<a.rows; l++)
        for (int c=0; c<a.cols; c++)
            if (a(l,c)==0) a(l,c)=255;
            else a(l,c)=0;
    imp(a,"inv_cek.pgm");
}