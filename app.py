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

@app.route('/results', methods=['POST'])
def predict_clickbait_youtube():
    data = [str(x) for x in request.form.values()]
    data = data[0]
    relevancy = scrape(data)
    clickbait_score = clickbait_predictor_yt.predict(data)
    filestr = Image.open(request.files['filename'])
    (nsfw_str, nsfw_score) = predict_nsfw(filestr)
    final_score = score(clickbait_score, nsfw_score, relevancy)
    clickbait_score = round(clickbait_score * 100)
    if clickbait_score == 0:
        clickbait_score = 1
    nsfw_score = round(nsfw_score * 100)
    if nsfw_score == 0:
        nsfw_score = 1
    print("Clickbait: " + str(clickbait_score) + ", NSFW: " + str(nsfw_score))
    return render_template('prediction.html', prediction_clickbait_youtube_text=clickbait_score, prediction_nsfw_text = nsfw_score, relevancy_results=relevancy, virality_score=final_score)

# @app.route('/predict_nsfw',methods=['POST'])
def predict_nsfw(filestr):
    # get data
    # filestr = Image.open(request.files['filename'])

    
    #image = np.fromstring(r.data, np.uint8)

    # data = request.get_json(force=True)
    # data = np.array(data)

    data = np.array(filestr)

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

    # # output = {'new_predictions': new_predictions.tolist()}
    # # return jsonify(results=output)
    # return render_template('index.html', prediction_text='Results Are:  $ {}'.format(predictions[0]))
    nsfw_score = '\tSFW score:\t{}\n\tNSFW score:\t{}'.format(*predictions[0])
    return (nsfw_score, predictions[0][1])
    # return render_template('index.html', prediction_nsfw_text='\tSFW score:\t{}\n\tNSFW score:\t{}'.format(*predictions[0]))

def scrape(title):
    # get data
    start_time = time.time()
    extractor = Extract(title)
    keywords = extractor.extract_keywords()
    p1 = Relevancy_Scraper(keywords[0])
    p1_tot_views = p1.get_adj_views()
    p1.close()
    result = p1.to_string(p1_tot_views)
    relevancy = [result]
    x = time.time() - start_time
    if x > 16.0:
        print("Time for scraping: " + str(x))
        return relevancy
    if len(keywords) > 1:
        p2 = Relevancy_Scraper(keywords[1])
        p2_tot_views = p2.get_adj_views()
        p2.close()
        result2 = p2.to_string(p2_tot_views)
        # tot_keywords = keywords[0] + " " + keywords[1]
        # p3 = Relevancy_Scraper(tot_keywords)
        # p3_tot_views = p3.get_adj_views()
        # p3.close()
        # result3 = p3.to_string(p3_tot_views)
        relevancy = [result, result2]
    x = time.time() - start_time
    print("Time for scraping: " + str(x))
    return relevancy
    # return render_template('index.html', relevancy_results=result)

def score(var1, var2, var3):
    clickbait = int(var1 * 10)
    nsfw = int(var2 * 10)
    lst = [5,0]
    for i in range(len(var3)):
        if "Irrelevant" in var3[i]:
            lst[i] = 2
        elif "Niche" in var3[i]:
            lst[i] = 4
        elif "Relevant" in var3[i]:
            lst[i] = 7
        elif "Popular" in var3[i]:
            lst[i] = 10
    avg = sum(lst) / len(var3)
    clickbait_weighted = clickbait * 0.25
    nsfw_weighted = nsfw * 0.3
    avg = avg * 0.45
    return clickbait_weighted + nsfw_weighted + avg

    

if __name__ == '__main__':
    app.run(port = 5000, debug=True)