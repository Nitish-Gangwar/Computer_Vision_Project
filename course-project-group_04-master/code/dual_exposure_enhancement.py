import numpy as np
import cv2
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from fuse_images import *


###	function creates a gaussian filter 
def create_spatial_affinity_guassian_weight_filter(sigma,size: int = 15):
	sigma=float(sigma)
	size=int(size)
	filter_window=np.zeros((size,size))
	p2=np.array((int(size/2),int(size/2)))
	for i in range(0,size):
		for j in range(0,size):
			p1=np.array((i,j))
			filter_window[i,j]=np.exp((-0.5)*np.sqrt(np.sum(np.square(p1-p2)))/np.square(sigma))
	return filter_window
	

def compute_smoothness_weights(illuminated_values, x: int, filter_window, eps: float = 1e-3):
	###	uncomment this for sobel filter
	temp=int(x==1)
	illuminated_values_sobel=cv2.Sobel(illuminated_values,cv2.CV_64F,temp,1-temp,ksize=5)
	
	###	This portion is for detecting the edges by using 
	###	reference : https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	'''
	grayscale_blur = cv2.GaussianBlur(illuminated_values,(3,3),0)
	selected_sigma = 0.33
	median= np.median(grayscale_blur)
	lower = int(max(0, (1.0 - selected_sigma)* median))
	upper = int(min(255, (1.0 + selected_sigma) * median)) 
	grayscale_blur = np.uint8(grayscale_blur)
	illuminated_values_sobel = cv2.Canny(grayscale_blur,lower,upper)
	'''
	###	uncomment this portion to see output from sobel or canny edge detector
	'''
	cv2.imshow("sobel's magic",illuminated_values_sobel)
	cv2.waitKey(0)
	'''
	
	all_ones=np.ones_like(illuminated_values)
	
	T = convolve(all_ones, filter_window, mode='constant')
	
	T=T/(np.abs(convolve(illuminated_values_sobel,filter_window,mode='constant'))+eps)
	result=T/(np.abs(illuminated_values_sobel)+eps)
	return result

def get_sparse_neighbour(p,n,m):
	###	function for obtaining the sparse neighbours
	a=p//m
	b=p%m
	dic={}
	if(a-1>=0):
		dic[(a-1)*m + b]=(a-1,b,0)
	if(a+1 < n):
		dic[(a+1)*m + b] = (a+1,b,0)
	if(b-1>=0):
		dic[a*m + b-1]=(a,b-1,1)
	if(b+1 < m):
		dic[a*m + b + 1]=(a,b+1,1)
	return dic

###	Function which refines the illumination map L by using spsolve
###	followed by performing the gamma adjustment
def refine_illumination_map_using_LIME(illuminated_values, gamma, lamda, filter_window, eps: float = 1e-3):
	weight_x = compute_smoothness_weights(illuminated_values, 1, filter_window, eps)
	
	weight_y = compute_smoothness_weights(illuminated_values, 0, filter_window, eps)
	
	n,m = illuminated_values.shape
	
	illumination_1d = illuminated_values.copy().flatten()
	
	no_of_rows=[]
	no_of_columns=[]
	data=[]
	
	total_count=n*m
	
	for p in range(total_count):
		diag=0
		for q,(k,l,x) in get_sparse_neighbour(p,n,m).items():
			if(x):
				weight=weight_x[k,l]
			else:
				weight = weight_y[k,l]
			no_of_rows.append(p)
			no_of_columns.append(q)
			data.append(-weight)
			diag = diag + weight
		no_of_rows.append(p)
		no_of_columns.append(p)
		data.append(diag)
	
	F = csr_matrix((data,(no_of_rows,no_of_columns)),shape = (total_count,total_count))
	
	Id = diags([np.ones(total_count)],[0])
	
	A = lamda * F + Id
	
	L_refined = spsolve(csr_matrix(A),illumination_1d,permc_spec=None,use_umfpack=True).reshape((n,m))
	
	L_refined = np.clip(L_refined,eps,1)**gamma
	
	return L_refined

###	function for computing the underexposed corrected image
def underexposure_image_correction(image, gamma, lamda, filter_window, eps: float = 1e-3,flag_name=0):
	### finding maximum pixel value in channel for initial illumination computation
	maximum_in_rows = np.max(image, axis=-1)
	filename = str(flag_name)+".jpg"
	
	'''
	cv2.imshow(filename,maximum_in_rows)
	cv2.waitKey(0)
	'''
	###	call for performing the refinement
	refined_maximum_value_in_row = refine_illumination_map_using_LIME(maximum_in_rows, gamma, lamda, filter_window, eps)
	
	refined_3d = np.repeat(refined_maximum_value_in_row[..., None], 3,axis=-1)
	corrected_image = image/refined_3d
	return corrected_image
	
#### function for handling under and over exposure present inside the image 
#### and then fusing the corrected images
def enhance_dual_exposure_image(image,gamma,lamda,sigma:int=3,contrast:float=1,saturation:float=1,exposedness:float = 1,eps:float= 1e-3):
	image=np.array(image)
	
	### getting spatial guassian filter made
	filter_window = create_spatial_affinity_guassian_weight_filter(sigma)
	
	### Normalizing the image
	normalized_image=image.astype(float)/255.0
	
	### Correcting the underexposure image
	under_corrected_images = underexposure_image_correction(normalized_image, gamma, lamda, filter_window, eps,0)
	
	'''
	cv2.imshow("Under_exposure_correction",under_corrected_images)
	cv2.waitKey(0)
	'''
	
	### Inverting first to perform over exposure correction in the form of under exposure correction
	inv_normalized_image = 1 - normalized_image
	
	'''
	cv2.imshow("Inverted_image",inv_normalized_image)
	cv2.waitKey(0)
	'''
	###	obtaining the over exposed corrected part
	over_exposure_corrected = 1 - underexposure_image_correction(inv_normalized_image,gamma,lamda,filter_window,eps,1)
	'''
	cv2.imshow("Over_exposure_correction",over_exposure_corrected)
	cv2.waitKey(0)
	'''
	### fusion of two exposure corrected images along with normalized input image
	corrected_image = fuse_multi_exposure_images(normalized_image,under_corrected_images,over_exposure_corrected,contrast,saturation,exposedness)
	return np.clip(corrected_image*255,0,255).astype("uint8")
	
