from flask import Flask , render_template , request 
import tensorflow as tf 
from PIL import Image
import numpy as np
import os


app=Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model/CropDiseaseModel.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_size=(226,226)
class_labels=["Apple_scab",
"Apple Black rot",
"Apple Healthy",
"Corn (maize) Cercospora leaf spot",
"Corn (maize) Northern Leaf Blight",
"Corn (maize) Healthy",
"Potato Early blight",
"Potato Late blight",
"Potato Healthy",
"Tomato Septoria leaf spot",
"Tomato Late blight",
"Tomato healthy"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return 'No image uploaded'

    

    UPLOAD_FOLDER = r"static\uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath)
    img=img.resize((226,226))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # TFLite prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_labels[np.argmax(output_data)]
    
    # img_url = '/' + filepath.replace('\\', '/')
    # print(img_url)

    return render_template('index.html', prediction=predicted_class , image_url=filepath)


if __name__ == "__main__":
    app.run(debug=True , host="0.0.0.0" , port=5000)