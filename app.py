from flask import Flask, jsonify, request, render_template
import pickle

from models.convnets import ConvolutionalNet
from models.predictor import Predictor

# load model
nsfw = pickle.load(open('models/nsfw.pkl', 'rb'))
clickbait_predictor_yt = Predictor("models/youtube_detector.h5", "data/vocabulary_youtube_porn.txt")

import tensorflow as tf
import numpy as np
from model import OpenNsfwModel, InputType
from scrape import Extract, Relevancy_Scraper
import skimage
import skimage.io
from PIL import Image
from io import BytesIO
import time

VGG_MEAN = [104, 117, 123]

# app
app = Flask(__name__)

# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [str(x) for x in request.form.values()]
    data = data[0]
    (relevancy, new_predictions) = predict_clickbait_youtube(data)

    filestr = Image.open(request.files['filename'])
    data = np.array(filestr)
    nsfw_predictions = predict_nsfw(data)

    print(nsfw_predictions)
    print(relevancy)
    print(new_predictions)

    return render_template('index.html', prediction_nsfw_text='\tSFW score:\t{}\n\tNSFW score:\t{}'.format(*nsfw_predictions), prediction_clickbait_youtube_text='Results Are:  $ {}'.format(new_predictions), relevancy_results=relevancy)

def predict_clickbait_youtube(data):
    relevancy = scrape(data)
    relevancy = "HELLO"
    new_predictions = clickbait_predictor_yt.predict(data)
    return (relevancy, new_predictions)

def predict_nsfw(data):
    img_data = data
    im = Image.fromarray(img_data.astype('uint8'), 'RGB')

    if im.mode != "RGB":
        im = im.convert('RGB')

    imr = im.resize((256, 256), resample=Image.BILINEAR)

    fh_im = BytesIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)

    image = (skimage.img_as_float(skimage.io.imread(fh_im, as_gray=False))
                    .astype(np.float32))

    H, W, _ = image.shape
    h, w = (224, 224)

    h_off = max((H - h) // 2, 0)
    w_off = max((W - w) // 2, 0)
    image = image[h_off:h_off + h, w_off:w_off + w, :]

    # RGB to BGR
    image = image[:, :, :: -1]

    image = image.astype(np.float32, copy=False)
    image = image * 255.0
    image -= np.array(VGG_MEAN, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    with tf.compat.v1.Session() as sess:
        nsfw.build(weights_path='open_nsfw-weights.npy')

        sess.run(tf.compat.v1.global_variables_initializer())
        predictions = sess.run(nsfw.predictions, feed_dict={nsfw.input: image})

    return predictions[0]

def scrape(title):
    # get data
    extractor = Extract(title)
    keywords = extractor.extract_keywords()

    p1 = Relevancy_Scraper(keywords[0])
    p1_tot_views = p1.get_adj_views()
    p1.close()
    result = p1.to_string(p1_tot_views)
    relevancy = [result]
    if len(keywords) > 1:
        p2 = Relevancy_Scraper(keywords[1])
        p2_tot_views = p2.get_adj_views()
        p2.close()
        result2 = p2.to_string(p2_tot_views)
        tot_keywords = keywords[0] + " " + keywords[1]
        p3 = Relevancy_Scraper(tot_keywords)
        p3_tot_views = p3.get_adj_views()
        p3.close()
        result3 = p3.to_string(p3_tot_views)
        relevancy = [result, result2, result3]
    print("finished all")
    print(relevancy)

    return relevancy

if __name__ == '__main__':
    app.run(port = 5000, debug=True)