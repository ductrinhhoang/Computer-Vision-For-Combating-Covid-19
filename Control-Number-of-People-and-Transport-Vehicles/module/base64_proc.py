import cv2
import base64
from PIL import Image
import numpy as np
import io


def img_to_base64(img):
    cnt = cv2.imencode('.jpg', img)[1][:, 0]
    b64 = base64.encodestring(cnt).decode('utf-8')
    return b64


def base64_to_img(base64_string):
    imgdata = base64.b64decode(base64_string)
    bgr_image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(bgr_image), cv2.COLOR_BGR2RGB)
