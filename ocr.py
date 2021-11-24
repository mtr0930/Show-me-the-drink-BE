import io
import os
import cv2 as cv
from PIL import Image
from google.cloud import vision
from google.cloud.vision_v1 import types
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
import numpy as np
from set_vars import classes
from test import change_image

drink_info = dict()
drink_info['cider'] = 0
drink_info['coca'] = 1
drink_info['cola'] = 1
drink_info['aca-cola'] = 1
drink_info['fanta'] = 2
drink_info['환타'] = 2
drink_info['환타 오렌지향'] = 2
drink_info['milkis'] = 3
drink_info['밀키스'] = 3
drink_info['monster'] = 4
drink_info['몬스터'] = 4
drink_info['mountain'] = 5
drink_info['dew'] = 5
drink_info['beenzino'] = 5
drink_info['pepsi'] = 7
drink_info['demisoda'] = 8
drink_info['sprite'] = 9
drink_info['prite'] = 9
drink_info['toreta'] = 10
drink_info['토레타'] = 10
drink_info['welchs'] = 11


def ocr(img_path):
    client = vision.ImageAnnotatorClient()

    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()

    img = vision.Image(content=content)

    response = client.text_detection(image=img)
    labels = response.text_annotations

    # OCR 결과에서 음료명 찾기
    for label in labels:
        print("label description", label.description)
        if label.description in drink_info.keys():
            return drink_info[label.description]
        if label.description.lower() in drink_info.keys():
            return drink_info[label.description.lower()]
    else:
        return -1  # OCR 결과도 없는 경우 -1 반환

def with_model(model, path):
    img_path = path
    changed_image_path = change_image(img_path)

    img = image.load_img(changed_image_path, target_size=(150, 150))

    x = image.img_to_array(img)
    x = x / 255.  # 이미지 rescale
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    predict = model.predict(images, batch_size=4, verbose=0)
    score = tf.nn.softmax(predict[0])
    print()
    np.set_printoptions(precision=3, suppress=True)

    result = predict.argmax()

    # 결과가 none일 경우 OCR 실행
    if classes[result] == 'none':
        result = ocr(img_path)
        print("ocr requested")
        print(result)
        if result == -1:
            print("IMAGE NAME: ---------- OCR 결과 없음, 다른 사진 요청")

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(classes[result], 100 * np.max(score))
    )
    print("RESULT: {:7} !!!!!!!!!".format(classes[result]))
    return classes[result]




def ocr_prediction(path):

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\g_chokuho\\Downloads\\drinks-ocr-api-ba17cbabe064.json'

    model = load_model("./model", compile=False)
    predict_result = with_model(model, path=path)
    return predict_result

# ocr_prediction("C:\\Users\\g_chokuho\\PycharmProjects\\LiteTF\\upload\\java.text.SimpleDateFormat@5069d960_img (1).jpg")
