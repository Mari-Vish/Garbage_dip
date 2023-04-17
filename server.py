import io
import json
import numpy as np
import tensorflow as tf
import neural_model
from PIL import Image
from flask import Flask, jsonify, request
from tensorflow import keras

app = Flask(__name__)

model = keras.models.load_model('../model.h5')
model = model()
model.load_weights('model_weights.h5')

class_names = ['battery', 'glass', 'lamp', 'metal', 'paper', 'plastic', 'trash']

@app.route('/predict', methods=['POST'])
def predict():
    img = Image.open(io.BytesIO(request.get_data())).convert('RGB')
    img_arr = np.array(img)
    img_arr = img_arr.astype('float32') / 255.
    img_arr = tf.image.resize(img_arr, (256, 256))
    pred = model.predict(np.array([img_arr]))
    result = {
        'class_name': class_names[np.argmax(pred)],
        'confidence': float(np.max(pred))
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run()
