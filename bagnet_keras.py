import keras
import bagnets.keras
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
from keras import backend as K
K.image_data_format()
K.set_image_data_format("channels_first")
print(K.image_data_format())
model = bagnets.keras.bagnet16()
cap = cv2.VideoCapture(0) 


def make_prediction():
    while(True):
        ret,frame = cap.read()
        img = cv2.resize(frame, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        out = decode_predictions(preds, top=3)[0]
        output = str(out[0][1])
        output2 = str(out[1][1])
        output3 = str(out[0][2])

        print(output)

if __name__ == '__main__':
    make_prediction()

