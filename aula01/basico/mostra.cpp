// mostra.cpp: Mostra a imagem lenna.jpg na tela.
// Apertando espaço, aparece a lista de comandos disponíveis.
// mostra.cpp -grad2015

#include <cekeikon.h>
    int main() {
    Mat_<COR> a;
    le(a,"lenna.jpg");
    mostra(a);
}