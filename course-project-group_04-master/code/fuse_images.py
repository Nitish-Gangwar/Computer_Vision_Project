import numpy as np
import cv2
def fuse_multi_exposure_images(input_im: np.ndarray, under: np.ndarray, over: np.ndarray,c_wt: float = 1, s_wt: float = 1, e_wt: float = 1):
    im=[]
    for i in [input_im,under,over]:
    	im.append(np.clip(i * 255, 0, 255).astype("uint8"))
    clipped_images=im
    created_mertn = cv2.createMergeMertens(c_wt, s_wt,e_wt)
    fused_imgs = created_mertn.process(clipped_images)
    return fused_imgs
