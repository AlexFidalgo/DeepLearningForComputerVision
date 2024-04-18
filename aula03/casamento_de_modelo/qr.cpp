#include "procimagem.h"

// Function to mark regions in an image based on a correlation matrix
Mat_<Vec3f> markRegions(Mat_<float> image, Mat_<float> correlation, float threshold) {
    Mat_<Vec3f> markedImage;
    cvtColor(image, markedImage, COLOR_GRAY2BGR);

    // Iterate over each pixel in the correlation matrix
    for (int row = 0; row < correlation.rows; row++) {
        for (int col = 0; col < correlation.cols; col++) {
            // Check if the correlation value at the current pixel exceeds the threshold
            if (correlation(row, col) >= threshold) {
                // Draw a rectangle around the region
                rectangle(markedImage, Point(col - 25, row - 25), Point(col + 25, row + 25), Scalar(0.0, 0.0, 1.0), 3);
            }
        }
    }

    return markedImage;
}

int main() {
    // Read input image 'op00.jpg' and normalize
    Mat_<float> inputImage = imread("op00.jpg", 0) / 255.0;

    // Read template image 'padrao_reduz.png' and normalize
    Mat_<float> templateImage = imread("padrao_reduz.png", 0) / 255.0;

    // Preprocess the template image
    Mat_<float> preprocessedTemplate = somaAbsDois(dcReject(templateImage));

    // Perform correlation using preprocessed template and input image
    Mat_<float> correlation1 = filtro2d(inputImage, preprocessedTemplate);
    imwrite("qr-p1.png", 255.0 * correlation1);

    // Mark regions in the input image based on correlation result
    Mat_<Vec3f> markedImage1 = markRegions(inputImage, correlation1, 0.6);
    imwrite("qr-m1.png", 255.0 * markedImage1);

    // Perform correlation using preprocessed template and input image (method: TM_CCORR)
    Mat_<float> correlation2 = matchTemplateSame(inputImage, preprocessedTemplate, TM_CCORR);
    imwrite("qr-p2.png", 255.0 * correlation2);

    // Mark regions in the input image based on correlation result
    Mat_<Vec3f> markedImage2 = markRegions(inputImage, correlation2, 0.6);
    imwrite("qr-m2.png", 255.0 * markedImage2);

    // Perform correlation using original template and input image (method: TM_CCOEFF_NORMED)
    Mat_<float> correlation3 = matchTemplateSame(inputImage, templateImage, TM_CCOEFF_NORMED);
    imwrite("qr-p3.png", 255.0 * correlation3);

    // Mark regions in the input image based on correlation result
    Mat_<Vec3f> markedImage3 = markRegions(inputImage, correlation3, 0.6);
    imwrite("qr-m3.png", 255.0 * markedImage3);

    return 0;
}
