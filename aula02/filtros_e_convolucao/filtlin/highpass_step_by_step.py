import cv2
import numpy as np

def filtro2d(ent, ker, borderType=cv2.BORDER_DEFAULT):
  """
  Applies a 2D filter to an entry matrix.

  Args:
      ent: Entry matrix as a NumPy array (float32).
      ker: Kernel as a NumPy array (float32).
      borderType: OpenCV border handling mode (optional).

  Returns:
      Filtered matrix as a NumPy array (float32).
  """
  sai = cv2.filter2D(ent, cv2.CV_32F, ker, borderType=borderType)  # Use CV_32F for float32
  return sai

def main():
  """
  Prompts for manual entry of the entry matrix, applies a high-pass filter, and prints the result.
  """

  # Get dimensions of the entry matrix
  rows = int(input("Enter the number of rows in the entry matrix: "))
  cols = int(input("Enter the number of columns in the entry matrix: "))

  # Create empty matrix to store user input
  ent = np.zeros((rows, cols), dtype=np.float32)

  # Prompt user to enter values for the entry matrix
  print("Enter the values for the entry matrix (row by row):")
  for i in range(rows):
    row_values = map(float, input().split())  # Convert input string to float list
    ent[i] = np.array(list(row_values))  # Convert float list to NumPy array and assign to row

  # Define high-pass kernel
  ker = np.array([
      [1, 1, 1],
      [1, -8, 1],
      [1, 1, 1]
  ], dtype=np.float32)

  # Normalize the kernel
  ker /= 8.0

  # Apply the filter
  sai = filtro2d(ent, ker)

  # Print the filtered matrix
  print("Filtered matrix:")
  print(sai)

if __name__ == "__main__":
  main()
