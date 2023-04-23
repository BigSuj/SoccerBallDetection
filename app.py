from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import onnxruntime as ort


app = Flask(__name__)

# Load the trained model and class names

sess = ort.InferenceSession("_model.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
class_names = ['Not a soccer ball', 'soccer ball']  # list of class names

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((224, 224))
    image = image.convert('RGB')
    
    # Convert the image to a NumPy array and scale the pixel values to [0, 1]
    x = np.array(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    
    
    return x

@app.route('/', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        # Get the uploaded file from the form
        file = request.files['image']
        
        # Load the image and preprocess it
        img = Image.open(file)
        x = preprocess_image(img)
        
        # Make a prediction using the model
        y = sess.run([output_name], {input_name: x})
        class_idx = np.argmax(y)
        class_label = class_names[class_idx]
        
        # Render the results page with the predicted class label
        return render_template('results.html', class_label=class_label)
    else:
        # Render the form page to upload an imag
        return render_template('form.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
