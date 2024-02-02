import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def make_prediction(model, image_path):
    # read the image and preprocess it
    og_image = cv2.imread(image_path)
    image = cv2.resize(cv2.imread(image_path), (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # make a prediction
    labels = ['No Violence Detected', 'Violence Detected']
    predictions = model.predict(image)
    prediction = labels[np.argmax(predictions, axis=1)[0]]

    # return og image for graphing
    return prediction, og_image

def make_frame_prediction(model, frame):
    # preprocess frame
    frame = cv2.resize(frame, (128, 128))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    # make prediction
    predictions = model.predict(frame)
    prediction = np.argmax(predictions, axis=1)[0]
    return prediction

def graph_predictions(predictions, og_images):
    # create graph
    plotting_count = len(predictions)
    plt.figure(figsize=(2 * plotting_count, 5 * plotting_count))
    # plot each image and prediction
    for i, image in enumerate(og_images):
        plt.subplot(plotting_count, 1, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(predictions[i],fontsize=30)
    plt.savefig(os.path.join('output/','Predictions.png'))
    plt.close()