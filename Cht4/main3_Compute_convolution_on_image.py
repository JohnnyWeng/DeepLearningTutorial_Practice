# pip install opencv-python
import cv2
import skimage

# Get a built-in image from skimage
image = skimage.data.chelsea()
cv2.imshow("original", image)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# Press Enter to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
