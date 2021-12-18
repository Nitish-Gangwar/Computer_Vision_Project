import argparse
from os import makedirs
import os
from os.path import join, exists, dirname, basename
from dual_exposure_enhancement import *


###	This function takes video frame as input and they are processed one by one
def main(args):
    video = args.video
    video=os.path.abspath(video)
    cap = cv2.VideoCapture(video)
    i=0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second "+str(fps))
    sigma=3
    contrast=1
    saturation=1
    exposedness=1
    eps=1e-3
    out_arr=[]
    path = dirname(video)
    file_name = basename(video)
    directory = join(path, "output")

    if not exists(directory):
        makedirs(directory)
    
    ###	 loop to process video frame by frame
    while(cap.isOpened()):
    	ret, frame = cap.read()
    	if ret == False:
    		break
    	
    	enhanced_image = enhance_dual_exposure_image(frame,args.gamma,args.lamda,sigma,contrast,saturation,exposedness,eps)
    	out_arr.append(enhanced_image)
    	i=i+1
    	print("Frame being enhanched is ",i)
        #print(" enhancing frame number " + str(i))
    out_arr=np.array(out_arr)
    height, width, layers = out_arr[0].shape
    size = (width,height)
    cap.release()
    cv2.destroyAllWindows()

    out = cv2.VideoWriter(directory+'/'+file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(out_arr)):
        out.write(out_arr[i])
    out.release()


###	Main function for taking the parameter values from terminal
if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("--video",type=str)
	parser.add_argument("--gamma",default=0.6,type=float)
	parser.add_argument("--lamda",default=0.15,type=float)
	args = parser.parse_args()
	main(args)
