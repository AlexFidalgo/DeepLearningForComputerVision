#include <iostream>
#include <unordered_map>
#include <opencv2/core.hpp>

// Function to calculate the mode of a Mat_<float> input
float get_mode(cv::Mat_<float> v) {
    std::unordered_map<float, int> found_values;

    // Count occurrences of each element
    for (int i = 0; i < v.rows; ++i) {
        for (int j = 0; j < v.cols; ++j) {
            float elem = v(i, j);
            found_values[elem] = found_values[elem] + 1;
        }
    }

    // Find the mode (first mode encountered)
    float mode = 0;
    int max_count = 0;
    for (auto& pair : found_values) { // range-based for loop, introduced in C++11, used for iterating over the elements of a container. 
        if (pair.second > max_count) {
            mode = pair.first;
            max_count = pair.second;
        }
    }

    return mode;
}

int main() {
    // Create a sample Mat_<float> for testing
    cv::Mat_<float> sample_mat(3, 3);
    sample_mat << 1.1, 2.2, 3.3,
                  1.1, 2.2, 2.2,
                  3.3, 3.3, 3.3;

    // Call the get_mode function and print the result
    float mode = get_mode(sample_mat);
    std::cout << "Mode of the matrix: " << mode << std::endl;

    return 0;
}
