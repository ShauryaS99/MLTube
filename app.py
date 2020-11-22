from flask import Flask, jsonify, request, render_template
import pickle

# load model
svm = pickle.load(open('svm.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
nsfw = pickle.load(open('nsfw.pkl', 'rb'))

import tensorflow as tf
import numpy as np
from model import OpenNsfwModel, InputType
import skimage
import skimage.io
from PIL import Image
from io import BytesIO

VGG_MEAN = [104, 117, 123]

# app
app = Flask(__name__)

# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # get data
    data = [str(x) for x in request.form.values()]
    new_vectors = vectorizer.transform(data)
    new_predictions = svm.predict(new_vectors)
    
    print(new_predictions)
    output = ["Clickbait" if x=='1' else "Not Clickbait" for x in new_predictions]
    
    # output = {'new_predictions': new_predictions.tolist()}
    # return jsonify(results=output)
    return render_template('index.html', prediction_text='Results Are:  $ {}'.format(output))

@app.route('/predict_nsfw',methods=['POST'])
def predict_nsfw():
    # get data
    filestr = Image.open(request.files['filename'])

    
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

    with tf.Session() as sess:
        nsfw.build(weights_path='open_nsfw-weights.npy')

        sess.run(tf.global_variables_initializer())
        predictions = sess.run(nsfw.predictions, feed_dict={nsfw.input: image})
        print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))

    # # output = {'new_predictions': new_predictions.tolist()}
    # # return jsonify(results=output)
    # return render_template('index.html', prediction_text='Results Are:  $ {}'.format(predictions[0]))
    return render_template('index.html', prediction_nsfw_text='\tSFW score:\t{}\n\tNSFW score:\t{}'.format(*predictions[0]))

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
