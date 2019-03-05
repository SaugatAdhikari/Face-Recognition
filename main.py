from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randint

from flask import Flask, request, render_template, send_from_directory, jsonify


import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import pickle
import json
import time
import datetime
import math
from sklearn.svm import SVC
from attendence import *
from write_file import *


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            self.x = self.graph.get_tensor_by_name("Placeholder:0")
            self.hold_prob1 = self.graph.get_tensor_by_name("Placeholder_2:0")
            self.hold_prob2 = self.graph.get_tensor_by_name("Placeholder_3:0")
            self.y_pred = self.graph.get_tensor_by_name("add_7:0")


    def run(self, data):
        """ Running the activation operation previously imported """
        return self.sess.run([tf.nn.softmax(self.y_pred)], feed_dict={self.x: data, self.hold_prob1: 1, self.hold_prob2: 1})

model_1 = ImportGraph(os.getcwd() + '/models/liveness detection/model.ckpt')

@app.route("/")
def index():
    return render_template("upload_new.html")


#Upload new Faces
@app.route("/upload", methods=["POST"])
def upload():

    label = request.form['name']
    print("Name: ", label)

    id = request.form['id']

    a= store_id(label,id)
    a.write_json()
    raw_dir = os.getcwd() + '/datasets/raw/' + str(id)
    if not os.path.isdir(raw_dir) and a.make_dir:
        os.mkdir(raw_dir)


    else:
        print("Couldn't create upload directory: {}".format(raw_dir))


    for upload in request.files.getlist("file"):
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([raw_dir, filename])
        # print ("Accept incoming file:", filename)
        # print ("Save it to:", destination)
        # print("Destination: ....", destination)
        upload.save(destination)



    #Align Data
    output_dir_path = os.getcwd() + '/datasets/aligned'
    output_dir = os.path.expanduser(output_dir_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datadir = os.getcwd() + '/datasets/raw'
    dataset = facenet.get_dataset(datadir)

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.getcwd() + '/align')

    minsize = 40  # minimum size of face
    threshold = [0.6, 0.8, 0.92]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    image_size = 182

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    print('Goodluck')

    # with open(bounding_boxes_filename, "w") as text_file:

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                    print('read data dimension: ', img.ndim)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        # text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                        print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
                    print('after data dimension: ', img.ndim)

                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('detected_face: %d' % nrof_faces)
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                            img_center = img_size / 2
                            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                            index = np.argmax(
                                bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                            det = det[index, :]
                        det = np.squeeze(det)
                        bb_temp = np.zeros(4, dtype=np.int32)

                        bb_temp[0] = det[0]
                        bb_temp[1] = det[1]
                        bb_temp[2] = det[2]
                        bb_temp[3] = det[3]

                        cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                        print("Cropped: ", cropped_temp)
                        # scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')
                        scaled_temp = cv2.resize(cropped_temp, (image_size, image_size))
                        print("Scaled: ", scaled_temp)
                        # scaled_temp.tile = [e for e in scaled_temp.tile if e[1][2] < 2181 and e[1][3] < 1294]

                        nrof_successfully_aligned += 1
                        misc.imsave(output_filename, scaled_temp)
                        # text_file.write(
                        #     '%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                    else:
                        print('Unable to align "%s"' % image_path)
                        # text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    return jsonify(status="Uploaded")


#Train Data
@app.route("/train", methods=["POST"])
def train():

    with tf.Graph().as_default():

        with tf.Session() as sess:

            datadir = os.getcwd() + '/datasets/aligned'
            dataset = facenet.get_dataset(datadir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

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

            classifier_filename = os.getcwd() + '/models/classifier/classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)

            # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True, tol = 0.5)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]
            print("iam classname",class_names)
            # Saving classifier model

            open("train.txt", "w").writelines([l for l in open("data.txt").readlines()])

            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            print('Goodluck')

    return jsonify(status="Trained")


#Recognize
@app.route("/recognize", methods=["GET","POST"])
def recognize():

    for upload in request.files.getlist("file"):

        img_array = np.array(bytearray(upload.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():

                pnet, rnet, onet = detect_face.create_mtcnn(sess, os.getcwd() + '/align')

                minsize = 40  # minimum size of face
                threshold = [0.6, 0.8, 0.92]  # three steps's threshold
                factor = 0.709  # scale factor
                image_size = 182
                input_image_size = 160

                currTime = time.time()
                with open('train.txt') as json_file:
                    HumanNames = json.load(json_file)

                presentTime = time.time()
                print("Time",presentTime)
                print("Loading time ", presentTime - currTime)

                print('Loading feature extraction model')


                modeldir = os.getcwd() + '/models/facenet/20170512-110547.pb'
                facenet.load_model(modeldir)


                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                classifier_filename = os.getcwd() + '/models/classifier/classifier.pkl'
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                    print('load classifier file-> %s' % classifier_filename_exp)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(2, 2),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                print("Found {0} faces!".format(len(faces)))

                # Crop Padding
                left = 1
                right = 1
                top = 1
                bottom = 1

                nameList = []
                if len(faces) > 0:
                    # Draw a rectangle around the faces
                    for (p, y, w, h) in faces:
                        print(p, y, w, h)

                        # Dubugging boxes
                        cv2.rectangle(frame, (p, y), (p + w, p + h), (0, 255, 0), 2)

                    img = frame[y - top:y + h + bottom, p - left:p + w + right]
                    print("Img: ", img.shape)

                    if img.shape[2] == 4:
                        #convert the image from RGBA2RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    img_data = cv2.resize(img, (227, 227))
                    img = np.reshape(img_data, [-1, 227, 227, 3])
                    img = np.array(img)

                    #For Liveness Detection
                    # k = model_1.run(img)
                    # a = k[0]
                    # b = a[0]
                    #
                    # detection = np.round(k[0], 3).argmax()

                    print('Start Recognition!')

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)

                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Face Detected: %d' % nrof_faces)

                    if nrof_faces == 0:
                        name = "Unknown"
                        id = None
                        checkin_time = None
                        response = {"id": id, "name": name, "time": checkin_time}
                        nameList.append(response)

                    # nameList = []
                # if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)


                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is too close')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)

                        # print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        # print(best_class_indices)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        # if detection == 0:
                        l = list(HumanNames)
                        print(l)
                        l.sort()
                        print("sorted list: ", l)
                        try:
                            best_class = l[best_class_indices[0]]
                            print("Best class: ", best_class)
                        except:
                            print(best_class_indices[0])

                        result_names = HumanNames[best_class]
                        print("Result name: ", result_names)
                        print("ID: ", best_class)
                        print(best_class_probabilities)
                        if best_class_probabilities >= 0.7:
                            print(result_names)
                            print(best_class_probabilities)
                            name = result_names
                            id = int(best_class)
                            attendance_register(id)
                            now = datetime.datetime.now()
                            checkin_time = now.strftime("%H:%M:%S")
                        else:
                            name = "Unknown"
                            id = None
                            checkin_time = None


                        # if detection == 1:
                        #
                        #     name = "INTRUDER"
                        #     id = None
                        #     checkin_time = None
                        response = {"id":id,"name":name, "time":checkin_time}
                        nameList.append(response)


                else:
                    print('Unable to align')
                    name = "Unknown"
                    id = None
                    checkin_time = None
                    response = {"id":id,"name":name, "time":checkin_time}
                    nameList.append(response)

    return jsonify(response=nameList)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8021, debug=True)
