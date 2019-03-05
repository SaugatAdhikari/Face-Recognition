from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, render_template, send_from_directory, jsonify

import tensorflow as tf
import numpy as np
import facenet
import detect_face
import os
import math
import pickle
import json
from sklearn.svm import SVC

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/train", methods=["POST"])
def train():

    with tf.Graph().as_default():

        with tf.Session() as sess:

            datadir = os.getcwd() + '/datasets/aligned'
            dataset = facenet.get_dataset(datadir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # classnames = []
            # [classnames.append(paths[i].split('/')[-2]) for i in range(len(paths)) if not paths[i].split('/')[-2] in classnames]
            # with open('data.txt', 'w') as outfile:
            #     json.dump(classnames, outfile)

            print('Loading feature extraction model')
            modeldir = os.getcwd() + '/models/facenet/20170512-110547.pb'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            batch_size = 20
            image_size = 160
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename = os.getcwd() + '/models/classifier/mydata_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)

            # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True, tol = 0.5)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            print('Goodluck')

    return jsonify(status="Trained")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5556, debug=True)
