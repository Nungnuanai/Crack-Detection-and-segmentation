from keras.models import load_model
import cv2
import numba as np
from numba import jit, cuda
import csv

model_det = load_model("Model_new_5.h5")

model_det.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

from data import *
mode = load_model("crack96x96_4.hdf5")

h = 50
w = 50
pad = 0

'''
img1 = cv2.imread('b.jpg')
img = cv2.imread('b.jpg')
img2 = cv2.imread('b.jpg',0)
img3 = cv2.imread('b.jpg',0)
 
row = img.shape[0]  # row - hight
col = img.shape[1]  # column - width

'''
@jit
def detect(img1,img,img2,img3):
    #img = cv2.imread('b.jpg')

    #img3 = cv2.imread('b.jpg',0)
    crk = 0
    row = img.shape[0]  # row - hight
    col = img.shape[1]  # column - width
    for i in range(row):
        for j in range(col):
            if i < h and j < w:
                patch = img[0:i + h, 0:j + w]
            elif i < w and j >= h:
                patch = img[0:i + h, j - w:j + w]
            elif i >= w and j < w:
                patch = img[i - h:i + h, 0:j + w]
            else:
                patch = img[i - h:i + h, j - w:j + w]
            patch = cv2.resize(patch, (100, 100))
            patch = np.reshape(patch, [1, 100, 100, 3])
            weights = model_det.predict_proba(patch)
            acc = float(weights[:,1])

            if acc<0.93:
                crk = crk + 1
                img3[i,j] = 255
                cv2.imwrite('b_detect.jpg',img3)
    print(crk)
    for i in range(row):
        for j in range(col):
            if i <= h and j <= w:
                patch = img1[0:i + h, 0:j + w]
                patch1 = img2[0:i + h, 0:j + w]
                patch = cv2.resize(patch, (100, 100))
                patch = np.reshape(patch, [1, 100, 100, 3])
                weights = model_det.predict_proba(patch)
                acc = float(weights[:,1])
                if acc >= 0.93:
                    patch1 = cv2.resize(patch1, (96, 96))
                    name = "image/" + "0.jpg"
                    cv2.imwrite(name, patch1)
                    testGene = testGenerator("image")
                    results = mode.predict_generator(testGene,1, verbose=1)
                    saveResult("predict", results)
                    seg = cv2.imread("predict/0_predict.jpg",0)
                    seg = cv2.resize(seg, (j+w,i+h))
                    img3[0:i + h,0:j + w] = seg
                    cv2.imwrite('b_segment.jpg',img3)
            elif i < w and j > h:
                patch = img1[0:i + h, j - w:j + w]
                patch1 = img2[0:i + h, j - w:j + w]
                patch = cv2.resize(patch, (100, 100))
                patch = np.reshape(patch, [1, 100, 100, 3])
                weights = model_det.predict_proba(patch)
                acc = float(weights[:,1])
                if acc >= 0.93:
                    patch1 = cv2.resize(patch1, (96, 96))
                    name = "image/" + "0.jpg"
                    cv2.imwrite(name, patch1)
                    testGene = testGenerator("image")
                    results = mode.predict_generator(testGene,1, verbose=1)
                    saveResult("predict", results)
                    seg = cv2.imread("predict/0_predict.jpg",0)
                    seg = cv2.resize(seg, (2*w,i+h)) #moi sua
                    img3[0:i + h,j - w:j + w] = seg
                    cv2.imwrite('b_segment.jpg',img3)
            elif i > w and j < w:
                patch = img1[i - h:i + h, 0:j + w]
                patch1 = img2[i - h:i + h, 0:j + w]
                patch = cv2.resize(patch, (100, 100))
                patch = np.reshape(patch, [1, 100, 100, 3])
                weights = model_det.predict_proba(patch)
                acc = float(weights[:,1])
                if acc >= 0.93:
                    patch1 = cv2.resize(patch1, (96, 96))
                    name = "image/" + "0.jpg"
                    cv2.imwrite(name, patch1)
                    testGene = testGenerator("image")
                    results = mode.predict_generator(testGene,1, verbose=1)
                    saveResult("predict", results)
                    seg = cv2.imread("predict/0_predict.jpg",0)
                    seg = cv2.resize(seg, (j+w,2*w))
                    img3[i - h:i + h,0:j + w] = seg
                    cv2.imwrite('b_segment.jpg',img3)
            else:            # if i > w and j > h:
                patch = img1[i - h:i + h, j - w:j + w]
                patch1 = img2[i - h:i + h, j - w:j + w]
                patch = cv2.resize(patch, (100, 100))
                patch = np.reshape(patch, [1, 100, 100, 3])
                weights = model_det.predict_proba(patch)
                acc = float(weights[:,1])
                if acc >= 0.93:
                    patch1 = cv2.resize(patch1, (96, 96))
                    name = "image/" + "0.jpg"
                    cv2.imwrite(name, patch1)
                    testGene = testGenerator("image")
                    results = mode.predict_generator(testGene,1, verbose=1)
                    saveResult("predict", results)
                    seg = cv2.imread("predict/0_predict.jpg",0)
                    seg = cv2.resize(seg, (2*w,2*h))
                    img3[i - h:i + h,j - w:j + w] = seg
                    cv2.imwrite('b_segment.jpg',img3)
    return img3

imga = cv2.imread('b.jpg')
imgb = cv2.imread('b.jpg')
imgc = cv2.imread('b.jpg',0)
imgd = cv2.imread('b.jpg',0)
k = detect(imga,imgb,imgc,imgd)


