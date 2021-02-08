import numpy as np
import os
from flask import Flask, request, render_template, redirect
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = keras.models.load_model('model.h5')

app.config['ALLOWED_IMG_EXT'] = ["JPG", "JPEG", "GIF"]
# images is a set of img with all accepted extensions
app.config['UPLOADED_PHOTOS_DEST'] = 'D:/PythonProjects/ImageClassification/User_Test_Images'
app.config['ALLOWED_MAX_FILE_SIZE'] = 0.5 * 1024 * 1024


# 0.5MB max size is allowed

@app.route('/')
def home():
    return render_template('index.html')


def allowed_image(filename):

    # We only want files with a . in the filename
    if not "." in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_IMG_EXT"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["ALLOWED_MAX_FILE_SIZE"]:
        return True
    else:
        return False

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':

        if request.files:

            # if "filesize" in request.cookies:
            #     if not allowed_image_filesize(request.cookies["filesize"]):
            #         print("Filesize exceeded maximum limit")
            #         return render_template('index.html',prediction_text='file excedded max size')

            # img is the name of input type in html element
            get_image = request.files['image']
            filename = get_image.filename

            if filename == "":
                print("No filename")
                return render_template('index.html',prediction_text='file has no name')

            if not allowed_image(filename):
                print("Not allowed extension")
                return render_template('index.html',prediction_text='not allowed extension')

            filename = secure_filename(filename)

            path_to_img = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], filename)
            get_image.save(path_to_img)

            img = image.load_img(path_to_img, target_size=(150, 150))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            if classes[0] > 0:
                op = 'dog'
            else:
                op = 'cat'
            return render_template('index.html', prediction_text='Its a	'+op)



if __name__ == "__main__":
    app.run(debug=True)
