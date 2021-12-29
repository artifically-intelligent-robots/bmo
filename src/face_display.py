import cv2

path = '../resources/bmo-imgs/bmo9.jpg'
# list of bmo faces and expression equivalent
happy = cv2.imread(path)
cv2.imshow('happy', happy)

cv2.waitKey(0)

cv2.destroyAllWindows()


