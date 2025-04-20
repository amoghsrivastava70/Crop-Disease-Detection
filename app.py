from flask import Flask , render_template , request 
import requests
import tensorflow as tf 
from PIL import Image
import numpy as np
import os
import uuid
from gtts import gTTS


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
    
    img_url = '/' + filepath.replace('\\', '/')
    
    
    api_key="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"
    url="https://openrouter.ai/api/v1/chat/completions"
    
    headers = {"Authorization" : f"Bearer {api_key}" , "Content-Type":"application/json"}
    
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": f"Provide me details about the Crop Disease '{predicted_class}' with its cause and fix in 4-5 lines"}]
    }
    ai_res=""
    response=requests.post(url,headers=headers , json=data)
    print(response)
    if response.status_code==200:
        data_ret=response.json()["choices"][0]["message"]["content"].replace('**','')
        ai_res+=data_ret
    else:
        ai_res+="Sorry Unavailable"
    
    tts = gTTS(ai_res)
    audio_filename = f"static/audio/{uuid.uuid4()}.mp3"
    tts.save(audio_filename)

    return render_template('index.html', prediction=predicted_class , image_url=img_url , info=ai_res , audio_file=audio_filename)


if __name__ == "__main__":
    app.run(debug=True , host="0.0.0.0" , port=5000)