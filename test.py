import cv2 as cv
import mask_to_plot as msk
from numba import jit
import timeit

img_path="C:\\Users\\yadur\\Downloads\\Edge\\Dataset\\ADEChallengeData2016\\images\\validation\\"
mask_path="C:\\Users\\yadur\\Downloads\\Edge\\Dataset\\ADEChallengeData2016\\annotations\\validation\\"
overlay_path="C:\\Users\\yadur\\Downloads\\Edge\\Dataset\\ADEChallengeData2016\\Overlay\\validation\\"
base="ADE_val_"
def func():
	for i in range(10):
		num=str(i+1)
		while len(num)<8:
			num="0"+num;
		img_name=base+num
		img=cv.imread(img_path+img_name+".jpg")
		mask=cv.imread(mask_path+img_name+".png")
		contours=msk.give_contours(mask)
		newimg=msk.contours_overlay(contours,img)
		ret=cv.imwrite(overlay_path+img_name+".png",newimg)
		if ret:
			print(i)

func()
print("YES")
cv.waitKey(0);
cv.destroyAllWindows()