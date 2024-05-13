import cv2 

img = cv2.imread("datasets/images/001739.jpg")

cropped_image = img[163:317, 294:339]

cv2.imwrite("test.jpg", cropped_image)