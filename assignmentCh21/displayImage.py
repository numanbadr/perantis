import cv2 as cv

img = cv.imread(r'../samples/data/home.jpg')

cv.imshow('Image', img)

cv.waitKey(0)
cv.destroyAllWindows()
