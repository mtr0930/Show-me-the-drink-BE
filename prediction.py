import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model('./model')
img_height = 150
img_width = 150

def predict(data_path):

    img = image.load_img(data_path, target_size=(img_height, img_width))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.
    img_array = tf.expand_dims(img_array, axis=0)

    #predict함수가 결과예측해줌.
    predictions = model.predict(img_array)
    print(predictions[0][1])

    score = tf.nn.softmax(predictions[0])
    max_index = np.argmax(score)

    print(score)
    class_names = ["cider", "coke", "fanta", "milkis", "monster", "mtdew", "pepsi", "soda", "sprite", "toreta", "welchis", "none"]
    pr_result = class_names[max_index]
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return pr_result

