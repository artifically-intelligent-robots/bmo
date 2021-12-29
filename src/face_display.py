import sys
print(sys.path)
import cv2
import graphics.faces as face
# path = '../resources/bmo-imgs/bmo9.jpg'
# list of bmo faces and expression equivalent
happy = cv2.imread(face.happy)
cv2.imshow('happy', happy)
#
cv2.waitKey(0)
#
cv2.destroyAllWindows()
