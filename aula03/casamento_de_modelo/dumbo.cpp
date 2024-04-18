#include "procimagem.h"

int main() {
    // Read input image 'figurinhas.jpg' and normalize
    Mat_<float> inputImage = imread("figurinhas.jpg", 0) / 255.0;

    // Read template image 'dumbo.jpg' and normalize
    Mat_<float> templateImage = imread("dumbo.jpg", 0) / 255.0;

    // Preprocess the template image
    Mat_<float> preprocessedTemplate = somaAbsDois(dcReject(templateImage));

    // Perform correlation using preprocessed template and input image (method: TM_CCORR)
    Mat_<float> correlation1 = matchTemplateSame(inputImage, preprocessedTemplate, TM_CCORR);
    imwrite("dumbo_cc.pgm", 255.0 * correlation1);

    // Perform correlation using original template and input image (method: TM_CCOEFF_NORMED)
    Mat_<float> correlation2 = matchTemplateSame(inputImage, templateImage, TM_CCOEFF_NORMED);
    imwrite("dumbo_ncc.pgm", 255.0 * correlation2);

    return 0;
}
