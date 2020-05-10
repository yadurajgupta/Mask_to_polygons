import cv2 as cv
import mask_to_plot as msk
img_path="C:\\Users\\yadur\\Downloads\\Edge\\Dataset\\ADEChallengeData2016\\images\\training\\"
mask_path="C:\\Users\\yadur\\Downloads\\Edge\\Dataset\\ADEChallengeData2016\\annotations\\training\\"
img_name="ADE_train_00000010"

img=cv.imread(img_path+img_name+".jpg")
mask=cv.imread(mask_path+img_name+".png")

contours=msk.give_contours(mask)
newimg=msk.contours_overlay(contours,img)
msk.visualize_mask(mask)
cv.imshow('output',newimg)
cv.waitKey(0);
cv.destroyAllWindows()