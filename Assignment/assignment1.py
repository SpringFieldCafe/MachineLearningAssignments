import numpy as np
import cv2

Matrix=np.random.randint(0,256,(100,100),dtype=np.uint8)

array1=[]
array2=[]

for i in range(5):
    row_indices = np.random.permutation(Matrix.shape[0])
    mid = Matrix.shape[0] // 2

    part1_index0=row_indices[0:mid]
    part2_index0=row_indices[mid:]

    part1=Matrix[part1_index0,:]
    part2=Matrix[part2_index0,:]

    array1.append(part1)
    array2.append(part2)

for i in range(5):
    print(array1[i])
    cv2.imshow('1',array1[i])
    print(array2[i])
    cv2.imshow('2',array2[i])
    cv2.imshow('i',Matrix)
    cv2.waitKey()