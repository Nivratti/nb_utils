import os
import shutil
from pathlib import Path

import cv2
import PIL
import numpy as np

from operator import itemgetter

# from tqdm import tqdm  # progess bar @# console
from tqdm.notebook import tqdm
from tqdm.contrib.concurrent import process_map # multi-process - tqdm>=4.42.0

from psutil import cpu_count


def extract_faces_from_image(image_array, face_boxes, required_size=(160, 160), convert_2_numpy=True):
    """
    Extract face region from image array
    
    Args:
        image_array (array): Image pixels in numpy format
        face_boxes (list): list of face bounding boxes
        required_size (tuple, optional): Final image resolution. Defaults to (160, 160).
        convert_2_numpy (bool, optional): convert pil face image to numpy. Defaults to True.
    
    Returns:
        list: If convert_2_numpy flag set to true then it returns list of faces(face regions-roi) in numpy format 
              otherwise it returns faces in pillow format
    """
    face_images = []

    for face_box in face_boxes:
        # extract the bounding box from the first face
        x1, y1, width, height = face_box

        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        extracted_face_array = image_array[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(extracted_face_array)
        face_image = face_image.resize(required_size)

        if convert_2_numpy:
            face_array = asarray(face_image)
            face_images.append(face_array)
        else:
            face_images.append(face_image)

    return face_images

def highlight_faces(image, face_boxes, xy_max=True, is_confidence_available=True, width=3, outline_color="red", method="PIL"):
    """
    Highlight faces using pillow
    
    Args:
        image (object): either PiL image or numpy image object
        face_boxes (list): list of face boxes
        outline_color (str, optional): Border color. Defaults to "red".
    
    Returns:
        PIL: Image
    """
    if isinstance(image, np.ndarray):
        # pil_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
        image = PIL.Image.fromarray(image)

    draw = PIL.ImageDraw.Draw(image)

    # for each face, draw a rectangle based on coordinates
    for face_box in face_boxes:
        if is_confidence_available:
            if xy_max:
                # x_min, y_min, x_max, y_max
                x1, y1, x2, y2, confidence = face_box
            else:
                x1, y1, width, height, confidence = face_box
                x2, y2 = (x1 + width), (y1 + height)
        else:
            if xy_max:
                x1, y1, x2, y2 = face_box
            else:
                x1, y1, width, height = face_box
                x2, y2 = (x1 + width), (y1 + height)    

        rect_start = (x1, y1)
        rect_end = (x2, y2)
        draw.rectangle((rect_start, rect_end), width=width, outline=outline_color)
    return image

class CropImage:
    """
    Create patch from original input image by using bbox coordinate

    Usage:
        image_cropper = CropImage()

        image_bbox = face_box
        w_input = 224
        h_input = 224

        scale = 1.45

        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        patch_cropped_img = image_cropper.crop(**param)
        print(f"patch_cropped_img.shape : {patch_cropped_img.shape}")
        display(Image.fromarray(patch_cropped_img))
    """
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)

            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img


def save_out_image(out_image, destfile_abs_path):
    pil_out_img = Image.fromarray(out_image)

    # check is dest-subfolder exists
    if not os.path.exists(os.path.dirname(destfile_abs_path)):
        os.makedirs(os.path.dirname(destfile_abs_path), exist_ok=True)

    pil_out_img.save(destfile_abs_path)
    return destfile_abs_path

def extract_face_singlefile(params):
    source_dir = params["source_dir"]
    file_rel_path = params["file_rel_path"]
    dest_dir = params["dest_dir"]
    scale = params["scale"]
    scale_factor = params["scale_factor"]
    image_cropper = params["image_cropper"]
    resize = params["resize"]
    new_size = params["new_size"]
    save_resized_separatly = params["save_resized_separatly"]

    sourcefile_abs_path = os.path.join(source_dir, file_rel_path)

    log = {
        "sourcefile": file_rel_path,
    }

    # check is already done
    src_file_name, src_file_extension = os.path.splitext(sourcefile_abs_path)
    src_txt_filename = f"{src_file_name}_facebox_retina-mobilenet.txt"
    if not os.path.exists(src_txt_filename):
        log.update({
            "code": 100,
            "info": f"face detection result text file not found."
        })
        return log

    # Reading an image in default mode 
    image = cv2.imread(sourcefile_abs_path) 

    if image is None:
        log.update({
            "code": 101,
            "info": f"problem reading image, it's none."
        })
        return log

    # color conversion
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    try:
        # read from txt file -- if face deetction done already
        f = open(src_txt_filename, "r")
        lines = f.read().splitlines()
        f.close()

        if not lines:
            log.update({
                "code": 102,
                "info": f"problem reading facebox result line from txt file"
            })
            return log
            
        detection = lines[0].split(" ")
        face_box = [int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])]
        confidence = float(detection[4])
    except Exception as e:
        log.update({
            "code": 103,
            "info": f"either problem  with facebox file or may be wrong facebox result"
        })
        return log
        
    exceptions = []
    if face_box:
        # crop_img = img[y:y+h, x:x+w]
        x, y, w, h = face_box[0], face_box[1], face_box[2], face_box[3]

        if w < 120 or h < 120:
            log.update({
                "code": 104,
                "info": f"detected face box size less than threshold 120x120."
            })
            return log

        cropped_img = image_rgb[y:y+h, x:x+w]

        if cropped_img is None or cropped_img.shape[0] < 50 or cropped_img.shape[1] < 50:
            log.update({
                "code": 105,
                "info": f"cropped face is none or less than threshold."
            })
            return log

        if scale:
            w_input = new_size[0]
            h_input = new_size[1]

            param = {
                "org_img": image_rgb,
                "bbox": face_box,
                "scale": scale_factor,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            patch_cropped_img = image_cropper.crop(**param)
            scaled_resized_destfile_abs_path = os.path.join(dest_dir, file_rel_path)
            save_out_image(patch_cropped_img, scaled_resized_destfile_abs_path)
        else:
            if resize and new_size != (-1, -1):
                resized_img = cv2.resize(cropped_img, new_size)
                if save_resized_separatly:
                    resized_destfile_abs_path = os.path.join(separate_dir_save_resized, file_rel_path)
                    save_out_image(resized_img, resized_destfile_abs_path)
                else:
                    resized_destfile_abs_path = os.path.join(dest_dir, file_rel_path)
                    save_out_image(resized_img, resized_destfile_abs_path)
        
        if resize == False or save_resized_separatly == True:
            destfile_abs_path = os.path.join(dest_dir, file_rel_path)
            save_out_image(cropped_img, destfile_abs_path)

    return log


def extract_faces_recursive_fulldir_multiprocess(
    source_dir, dest_dir,
    scale=False, scale_factor=0.0, image_cropper=None,
    resize=False, new_size=(-1, -1), save_resized_separatly=False, separate_dir_save_resized="", 
    multiprocess=True, max_workers=5,
    ):
    """
    Extract faces and resize faces if required from images -- all
    Usage:
    source_dir  = "/content/train"
    dest_dir   = "/content/loose_cropped_face__resized_224x224__multi_scaled" # separate_dir_save_resized = "/content/tight_cropped_resized_face__224x224"


    results = do_multiscale_face_extraction(
        source_dir, 
        dest_dir,
        lst_scale_factor=[1, 1.5, 2, 2.7, 3, 4], 
        target_size=(224, 224), max_workers=5
    )
    """
    # list all files
    lst_files = list_files(source_dir, filter_ext=[".jpg", ".jpeg", ".png"], return_relative_path=True)

    lst_params = []
    for file_rel_path in lst_files:
        lst_params.append({
            "source_dir": source_dir,
            "file_rel_path": file_rel_path,
            "dest_dir": dest_dir,
            "scale": scale,
            "scale_factor": scale_factor,
            "image_cropper": image_cropper,
            "resize": resize,
            "new_size": new_size,
            "save_resized_separatly": save_resized_separatly,
        })

    max_workers = cpu_count(logical=True)

    # map multiple tasks
    result = process_map(extract_face_singlefile, lst_params , max_workers=max_workers)
    return result

def do_multiscale_face_extraction(source_dir, dest_dir, lst_scale_factor=[], target_size=(224, 224), max_workers=5):
    """ 
    Create multiscaled face cropping, resizing
    """
    # make out dir if not exists
    os.makedirs(dest_dir, exist_ok=True)

    image_cropper = CropImage()

    results = []
    for scale_factor in lst_scale_factor:
        new_out_foler = f"{scale_factor}_{target_size[0]}x{target_size[0]}" # ex. 3_224x224

        sub_dest_dir = os.path.join(dest_dir, new_out_foler)
        os.makedirs(sub_dest_dir, exist_ok=True)

        result = extract_faces_recusive_fulldir_multiprocess(
            source_dir,
            sub_dest_dir,
            scale=True,
            scale_factor=scale_factor,
            image_cropper=image_cropper,
            resize=True,
            new_size=target_size,
            save_resized_separatly=False,
            # separate_dir_save_resized=separate_dir_save_resized,
            multiprocess=True, max_workers=max_workers,
        )
        results.append(result)
    return results


def main():
    # resize only -- tight crop
    # source_dir  = "/content/train"
    # dest_dir   = "/content/tight_cropped_resized_face__224x224" # "/content/tight_cropped_face"
    # # separate_dir_save_resized = "/content/tight_cropped_resized_face__224x224"

    # # make out dir if not exists
    # os.makedirs(dest_dir, exist_ok=True)

    # extract_faces_recursive_fulldir_multiprocess(
    #     source_dir,
    #     dest_dir,
    #     resize=True,
    #     new_size=(224, 224),
    #     save_resized_separatly=False,
    #     # separate_dir_save_resized=separate_dir_save_resized,
    #     multiprocess=True, max_workers=5,
    # )


    ### with scaling -- loose face cropping
    source_dir  = "/content/train"
    dest_dir   = "/content/loose_cropped_face__resized_224x224__scaled_4" # separate_dir_save_resized = "/content/tight_cropped_resized_face__224x224"

    # make out dir if not exists
    os.makedirs(dest_dir, exist_ok=True)

    image_cropper = CropImage()

    result = extract_faces_recursive_fulldir_multiprocess(
        source_dir,
        dest_dir,
        scale=True,
        scale_factor=2,
        image_cropper=image_cropper,
        resize=True,
        new_size=(224, 224),
        save_resized_separatly=False,
        # separate_dir_save_resized=separate_dir_save_resized,
        multiprocess=True, max_workers=5,
    )