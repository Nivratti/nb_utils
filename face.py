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