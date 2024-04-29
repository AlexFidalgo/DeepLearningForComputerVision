#include "../procimagem.h"

int main() {
    // Load the original MNIST dataset
    MNIST mnist(11, true, true);
    mnist.le("/home/alex/cekeikon5/tiny_dnn/data");

    // Define the number of shifts
    const int num_shifts = 4;

    // Create a new matrix to hold the shifted images
    Mat shifted_ax;
    Mat original_ax = mnist.ax.clone(); // Clone the original dataset

    // Loop through each shift direction
    for (int shift = 0; shift < num_shifts; ++shift) {
        // Define the shift direction
        int dx = 0, dy = 0;
        if (shift == 0) dx = -1; // Shift left
        else if (shift == 1) dx = 1; // Shift right
        else if (shift == 2) dy = -1; // Shift up
        else if (shift == 3) dy = 1; // Shift down

        // Shift the images
        Mat shifted_images(original_ax.rows, original_ax.cols, original_ax.type());
        for (int i = 0; i < original_ax.rows; ++i) {
            Mat original_image = original_ax.row(i).reshape(1, 28); // Reshape to 28x28
            Mat shifted_image = Mat::zeros(original_image.size(), original_image.type());
            // Perform the shift
            Rect roi(0, 0, original_image.cols, original_image.rows);
            if (dx < 0) roi.x -= dx;
            if (dy < 0) roi.y -= dy;
            original_image(roi).copyTo(shifted_image(Rect(abs(dx), abs(dy), roi.width - abs(dx), roi.height - abs(dy))));
            shifted_images.row(i) = shifted_image.reshape(1, 1); // Reshape back to 1D and assign to the new matrix
        }

        // Append the shifted images to the dataset
        if (shifted_ax.empty())
            shifted_ax = shifted_images;
        else
            vconcat(shifted_ax, shifted_images, shifted_ax);
    }

    // Concatenate the original and shifted datasets
    vconcat(mnist.ax, shifted_ax, mnist.ax);

    // Measure time for training
    double t1 = tempo();

    // Train the classifier
    Ptr<ml::KNearest>  knn(ml::KNearest::create());
    knn->train(mnist.ax, ml::ROW_SAMPLE, mnist.ay);

    // Measure time for training completion
    double t2 = tempo();

    // Perform prediction
    Mat_<float> dist;
    knn->findNearest(mnist.qx, 1, noArray(), mnist.qp, dist);

    // Measure time for prediction
    double t3 = tempo();

    // Print results
    printf("Errors=%10.2f%%\n", 100.0 * mnist.contaErros() / mnist.nq);
    printf("Training Time: %f\n", t2 - t1);
    printf("Prediction Time: %f\n", t3 - t2);

    return 0;
}
