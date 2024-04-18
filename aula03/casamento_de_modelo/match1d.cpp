//match1d.cpp 2024
#include "procimagem.h"

int main()
{
    // Define input matrix 'a'
    Mat_<float> a = (Mat_<float>(1, 13) << 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0);
    cout << "Matrix a: " << endl << a << endl;

    // Define template matrix 'q'
    Mat_<float> q = (Mat_<float>(1, 3) << 0, 1, 1);
    cout << "Template q: " << endl << q << endl;

    // Apply filtro2d operation on matrix 'a' using template 'q'
    Mat_<float> p = filtro2d(a, q, BORDER_REPLICATE);
    cout << "Result of filtro2d operation with matrix a and template q: " << endl << p << endl;

    // Apply dcReject operation on template 'q'
    Mat_<float> q2 = dcReject(q);
    cout << "Template q after dcReject operation: " << endl << q2 << endl;

    // Apply filtro2d operation on matrix 'a' using dcRejected template 'q2'
    Mat_<float> p2 = filtro2d(a, q2, BORDER_REPLICATE);
    cout << "Result of filtro2d operation with matrix a and dcRejected template q2: " << endl << p2 << endl;

    // Add 10 to matrix 'a'
    Mat_<float> a3 = a + 10;
    cout << "Matrix a after adding 10 to each element: " << endl << a3 << endl;

    // Apply filtro2d operation on modified matrix 'a3' using dcRejected template 'q2'
    Mat_<float> p3 = filtro2d(a3, q2, BORDER_REPLICATE);
    cout << "Result of filtro2d operation with modified matrix a3 and dcRejected template q2: " << endl << p3 << endl;

    // Apply somaAbsDois operation on dcRejected template 'q'
    Mat_<float> q4 = somaAbsDois(dcReject(q));
    cout << "Template q after somaAbsDois operation on dcRejected q: " << endl << q4 << endl;

    // Apply filtro2d operation on matrix 'a' using modified template 'q4'
    Mat_<float> p4 = filtro2d(a, q4, BORDER_REPLICATE);
    cout << "Result of filtro2d operation with matrix a and modified template q4: " << endl << p4 << endl;

    return 0;
}
