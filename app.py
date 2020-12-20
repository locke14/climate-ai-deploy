import os

from flask import Flask, render_template, redirect, url_for, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from wtforms import SubmitField
from tensorflow.keras import backend as k
from tensorflow.keras.models import load_model
import numpy as np

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']

# evaluator = ModelEvaluator(os.path.join(basedir, 'model.h5'), (1, 40, 32, 3), 'rgb')

CLASSES = [('No-Anomaly',
            ' Nominal solar module',
            'static/no-anomaly.png'),
           ('Cell',
            'Hot spot occurring with square geometry in single cell',
            'static/cell.png'),
           ('Cell-Multi',
            'Hot spots occurring with square geometry in multiple cells',
            'static/cell-multi.png'),
           ('Cracking',
            'Module anomaly caused by cracking on module surface',
            'static/cracking.png'),
           ('Hot-Spot',
            'Hot spot on a thin film module',
            'static/hot-spot.png'),
           ('Hot-Spot-Multi',
            'Multiple hot spots on a thin film module',
            'static/hot-spot-multi.png'),
           ('Shadowing',
            'Sunlight obstructed by vegetation, man-made structures, or adjacent rows',
            'static/shadowing.png'),
           ('Diode',
            'Activated bypass diode,'
            ' typically 1/3 of module',
            'static/diode.png'),
           ('Diode-Multi',
            'Multiple activated bypass diodes, typically affecting 2/3 of module',
            'static/diode-multi.png'),
           ('Vegetation',
            'Panels blocked by vegetation',
            'static/vegetation.png'),
           ('Soiling',
            'Dirt, dust, or other debris on surface of module',
            'static/soiling.png'),
           ('Offline-Module',
            'Entire module is heated',
            'static/offline-module.png')]

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)


def recall(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + k.epsilon())


def precision(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + k.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + k.epsilon()))


try:
    with tf.device('/CPU:0'):
        model = load_model('trained_model.h5',
                           custom_objects={'f1': f1,
                                           'recall': recall,
                                           'precision': precision})
    err = 'Model loaded successfully'
except OSError as e:
    model = None
    err = f'Model load failed: {e}<br/>.' \
          f'Search directory contents: {os.listdir(basedir)}'


def predict(input_file):
    idx_to_class = {0: 'Cell',
                    1: 'Cell-Multi',
                    2: 'Cracking',
                    3: 'Diode',
                    4: 'Diode-Multi',
                    5: 'Hot-Spot',
                    6: 'Hot-Spot-Multi',
                    7: 'No-Anomaly',
                    8: 'Offline-Module',
                    9: 'Shadowing',
                    10: 'Soiling',
                    11: 'Vegetation'}

    im = load_img(input_file, color_mode='rgb')
    arr = img_to_array(im).reshape((1, 40, 32, 3))
    idx = np.argmax(model.predict(arr), axis=-1)
    return idx_to_class[idx[0]]


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file!')])
    submit = SubmitField('Upload')


@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if request.method == 'POST':
        request.files['file'].save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], request.files['file'].filename))
        return redirect(url_for('upload_file'))

    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    file_urls = [photos.url(filename) for filename in files_list]

    if model:
        predictions = [predict(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], file_name)) for file_name in files_list]
    else:
        predictions = ['Unknown'] * len(files_list)

    return render_template('index.html',
                           classes=CLASSES,
                           form=form,
                           files=zip(files_list, file_urls, predictions),
                           err=err)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = photos.path(filename)
    os.remove(file_path)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
