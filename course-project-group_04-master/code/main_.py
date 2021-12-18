import argparse
import numpy as np
from argparse import RawTextHelpFormatter
import glob
from os import makedirs
from os.path import join, exists, basename, splitext
import cv2
import os
from tqdm import tqdm
from dual_exposure_enhancement import *



def main(args):
    imdir = args.path
    image_formats = ['png', 'jpg','jpeg', 'bmp','dng']
    
    path=os.path.abspath(imdir)
    
    files=[]
    [files.extend(glob.glob(path + '/*.' + e)) for e in image_formats]

    images_collection=[]
    for filename in files:
    	inp_img=cv2.imread(filename)
    	images_collection.append(inp_img)
    '''
    cv2.imshow("input image",images_collection[0])
    cv2.waitKey(0)
    '''
    
    #print("path= ",os.path.abspath(path))
    directory = join(path, "output")

    if not exists(directory):
        makedirs(directory)
    
    sigma=3
    contrast=1
    saturation=1
    exposedness=1
    eps=1e-3
    for i, image in tqdm(enumerate(images_collection), desc="Enhancing images"):
    	cv2.imshow("original image",image)
    	cv2.waitKey(0)
    	
    	enhanced_image = enhance_dual_exposure_image(image, args.gamma, args.lamda,sigma, contrast, saturation, exposedness, eps)
    	corrected_name=basename(files[i])
    	name_=corrected_name.split(".")
    	first_name=name_[0]
    	corrected_name=first_name+str(".jpg")
    	print("corrected_name = ",corrected_name)
    	cv2.imwrite(join(directory, corrected_name), enhanced_image)
    	cv2.imshow("exposure corrected",enhanced_image)
    	cv2.waitKey(0)
    	cv2.destroyAllWindows()
    	
if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("--path",type=str)
	parser.add_argument("--gamma",default=0.6,type=float)
	parser.add_argument("--lamda",default=0.15,type=float)
	args = parser.parse_args()
	main(args)
	
#References:

#	1.https://www.geeksforgeeks.org/python-call-function-from-another-file/
#	2.https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
#	3.https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
#	4.https://github.com/pvnieo/Low-light-Image-Enhancement

