"""Main application file"""
from flask import Flask
import tensorflow as tf
app = Flask(__name__)

@app.route('/<image_classification>')
def model_prediction(new_image):
    """Load the tensorflow CNN model and return the prediction on new image"""
    loaded_model = tf.keras.models.load_model('./model.h5')
    prob = loaded_model.predict()
    return [0 if prob[1][0] > prob[1][1] else 1]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
