import cv2
import os

current_directory = os.path.abspath(os.path.dirname(__file__))

def are_images_equal(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    # Check if images are loaded successfully
    if image1 is None or image2 is None:
        print("Error: One or both images could not be loaded.")
        return False
    
    # Check if images have the same shape
    if image1.shape != image2.shape:
        print("Error: Images have different sizes.")
        return False
    
    # Check if all pixel values are equal
    if (image1 == image2).all():
        print("Images are equal.")
        return True
    else:
        print("Images are not equal.")
        return False

# Example usage
image1_path = "qpnn_cpp.pgm"
image1_path = os.path.join(current_directory, image1_path)
image2_path = "x_sklearn.pgm"
image2_path = os.path.join(current_directory, image2_path)
are_images_equal(image1_path, image2_path)
