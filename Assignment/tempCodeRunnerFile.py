import numpy as np
import cv2

Matrix=np.random.randint(0,256,(400,400))
cv2.imshow('i',Matrix)
cv2.waitKey()