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

def highlight_faces(pil_image, face_boxes, outline_color="red"):
    """
    Highlight faces using pillow
    
    Args:
        pil_image (object): PiL image object
        face_boxes (list): list of face boxes
        outline_color (str, optional): Border color. Defaults to "red".
    
    Returns:
        PIL: Image
    """
    draw = ImageDraw.Draw(pil_image)
    # for each face, draw a rectangle based on coordinates
    for face_box in face_boxes:
        x, y, width, height = face_box
        rect_start = (x, y)
        rect_end = ((x + width), (y + height))
        draw.rectangle((rect_start, rect_end), outline=outline_color)
    return pil_image


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