#include <iostream>
#include <algorithm>
#include <vector>

int main() {
    std::vector<int> vec = {9, 7, 2, 5, 4, 1, 8, 6, 3};

    // Print original vector
    std::cout << "Original vector: ";
    for (int num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Find the median (middle) element
    auto meio = vec.begin() + vec.size() / 2;

    // Partially sort the vector such that the median element is in the correct position
    std::nth_element(vec.begin(), meio, vec.end());

    // Print the partially sorted vector
    std::cout << "Partially sorted vector: ";
    for (int num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Print the median element
    std::cout << "Median: " << *meio << std::endl;

    return 0;
}
