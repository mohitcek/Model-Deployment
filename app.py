"""Main application file"""
from flask import Flask
import tensorflow as tf
from model_architecture import create_model
app = Flask(__name__)

@app.route('/<image_classification>')
def model_prediction(new_image):
    """Load the tensorflow CNN model and return the prediction on new image"""
    
    img_size = 224
    img_arr = cv2.imread(new_image)[...,::-1] #convert BGR to RGB format
    resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
    
    model_input = np.array(resized_arr).reshape(-1, img_size, img_size, 1)
    model_input = model_input/255

    model = create_model()
    model.load_weights('./checkpoints/my_checkpoint')
    prob = model.predict(model_input)
    return [0 if prob[1][0] > prob[1][1] else 1]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
