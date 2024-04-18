#include "procimagem.h"

int main() {
    // Define input matrix 'a'
    Mat_<float> a = (Mat_<float>(1, 13) << 0, 1, 5, 3, 1, -1, 3, 1, 1, -2, 6, 2, 0);
    cout << "Matrix a: " << endl << a << endl;

    // Define template matrix 'q'
    Mat_<float> q = (Mat_<float>(1, 3) << 0, 1, 0.5);
    cout << "Template q: " << endl << q << endl;

    // Apply somaAbsDois and dcReject operations on template 'q'
    Mat_<float> q2 = somaAbsDois(dcReject(q));
    cout << "Processed template q: " << endl << q2 << endl;

    // Perform template matching using processed template 'q2' and method TM_CCORR
    Mat_<float> p2 = matchTemplateSame(a, q2, TM_CCORR);
    cout << "Result of template matching with processed template q2 (method: TM_CCORR): " << endl << p2 << endl;

    // Perform template matching using original template 'q' and method TM_CCOEFF_NORMED
    Mat_<float> p3 = matchTemplateSame(a, q, TM_CCOEFF_NORMED);
    cout << "Result of template matching with original template q (method: TM_CCOEFF_NORMED): " << endl << p3 << endl;

    return 0;
}
