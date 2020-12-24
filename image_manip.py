import os, random
import shutil

import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np

def read_image(image_path, method="PIL", colorspace='RGB'):
    """
    Read image from disk
    
    Args:
        image_path (str): Image path
        method (str, optional): which library to use. Defaults to "PIL".
        rgb (bool, optional): convert to rgb. Defaults to True.
    
    Returns:
        tuple: image_object, and image pixels in numpy format
    """
    if method == "PIL":
        # load image from file
        image = Image.open(image_path)
        if colorspace == 'RGB':
            image = image.convert('RGB')
        elif colorspace == 'GRAY':
            image = image.convert('L')

        # convert to array
        # obtain_image_pixels
        pixels = np.asarray(image)
        return (image, pixels)
    elif method == "CV2":
        image_array = read_image_cv2(image_path, colorspace=colorspace)
        return (image_array, [])
    elif method == "MATPLOT":
        print("Not implimented yet")
        return (False, False)
    else:
        print("Please select proper reading method")
        return (False, False)
