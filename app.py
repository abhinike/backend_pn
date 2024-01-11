from flask import Flask, request, jsonify
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from PIL import Image

from io import BytesIO
import requests
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)



# Load the pre-trained model
with open('Mode_rfl.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def download_image_from_url(image_url):
    response = requests.get(image_url)
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print(f"Failed to download image from URL. Status code: {response.status_code}")
        return None
    




    # Load the Keras model
    model = tf.keras.models.load_model(filepath=model_path)

    # Download and save the image to a local path
    img = download_image_from_url(img_url)
    if img is not None:
        local_image_path = 'downloaded_image.jpg'  # Change this to your desired local path
        img.save(local_image_path)

        # Read the saved image using OpenCV
        # img_array = cv2.imread(local_image_path)

        img = img_to_array(load_img("downloaded_image.jpg", target_size=(150,150,3)))
        img = img * (1./255)

        VizGradCAM(model,img, plot_results=True, interpolant=0.5)
        img = np.expand_dims(img, axis=0)
        pred= model.predict(img)
        print(pred)

        # Upload the heatmap image directly to Firebase Storage
        heatmap_url = upload_heatmap_to_storage(load_img("downloaded_image.jpg", target_size=(150,150,3)))

        # Delete the locally saved image
        os.remove(local_image_path)

        return heatmap_url
        

    return None








def predict_pneumonia(img_url):
    img = download_image_from_url(img_url)
    
    if img is not None:
        img_array = img_to_array(img)
        
        # Check the number of channels in the image
        num_channels = img_array.shape[-1]
        
        if num_channels == 1:
            # Image is already grayscale
            temp = cv.resize(img_array[:, :, 0], (64, 64))
        else:
            # Convert to grayscale
            temp = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
        
        scaled = np.array(temp) / 255
        scaled_1d = scaled.reshape(1, 64 * 64)
        df_test = pd.DataFrame(scaled_1d)
        
        pred = loaded_model.predict(df_test)
        
        if pred == 0:
            return "Bacterial Pneumonia"
        elif pred == 1:
            return "Healthy Lung"
        else:
            return "Viral Pneumonia"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        img_url = data['image_url']

        # Get the prediction result
        result = predict_pneumonia(img_url)
        print(result)

        # Generate the heatmap and get the heatmap URL
       

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
