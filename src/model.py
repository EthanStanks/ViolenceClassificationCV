from keras.layers import Flatten, Dense, Dropout, Input
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from keras import models

BATCH_SIZE = 32
EPOCHS = 20

def preprocess():
    # location of images
    violence_dir = os.path.join('data/','train_images/','violence/')
    nonviolence_dir = os.path.join('data/','train_images/','non_violence/')

    violence_images = []
    violence_labels = []
    nonviolence_images = []
    nonviolence_labels = []

    # read and reize the images
    for file in os.listdir(violence_dir):
        image_path = os.path.join(violence_dir, file)
        violence_images.append(cv2.resize(cv2.imread(image_path),(128,128)))
        violence_labels.append(1)

    for file in os.listdir(nonviolence_dir):
        image_path = os.path.join(nonviolence_dir, file)
        nonviolence_images.append(cv2.resize(cv2.imread(image_path),(128,128)))
        nonviolence_labels.append(0)

    # combine images into one container
    images = violence_images + nonviolence_images
    labels = violence_labels + nonviolence_labels

    # numpy this, numpy that
    images = np.array(images)
    labels = np.array(labels)

    # split data for training
    xtrain, xtest, ytrain, ytest = train_test_split(images, labels, test_size=0.2, random_state=212024)
    xtrain, xtest = xtrain / 255.0, xtest / 255.0
    ytrain, ytest = ytrain.flatten(), ytest.flatten()
    return xtrain, ytrain, xtest, ytest

def define_model(xtrain):
    # network structure
    num_classes = 2
    i = Input(shape=xtrain[0].shape)

    # Conv_1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1,1))(i)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), strides=(2,2))(x)
    x = BatchNormalization()(x)
    
    # Conv_2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=(2,2))(x)
    x = BatchNormalization()(x)
    
    # Conv_3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2,2))(x)
    x = BatchNormalization()(x)

    # dense layer (FC)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(i, x)

def train_model(model, xtrain, ytrain, xtest, ytest):
    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    steps_per_epoch = xtrain.shape[0] // BATCH_SIZE
    history = model.fit(xtrain,ytrain, validation_data=[xtest, ytest],
                        steps_per_epoch=steps_per_epoch,
                        epochs=EPOCHS)
    
    # validate the accuracy and loss
    validate_model(history, model, xtrain, ytrain, xtest, ytest)

    # save the model
    save_model(model)

def validate_model(history, model, xtrain, ytrain, xtest, ytest):
    # plot accuracy and loss
    plot_scores(history)

    # evaluate train and test data
    test_loss, test_acc = model.evaluate(xtest, ytest)
    train_loss, train_acc = model.evaluate(xtrain, ytrain)
    print(f"Test Loss: {round(test_loss,2)} | Test Accuracy: {round(test_acc,2)}")
    print(f"Train Loss: {round(train_loss,2)} | Train Accuracy: {round(train_acc,2)}")

def save_model(model):
    # save model
    model.save(os.path.join('output/','violence_vgg16.h5'))

def load_model():
    # load the saved model
    path = os.path.join('output/','violence_vgg16.h5')
    return models.load_model(path)


def plot_scores(history):
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.savefig(os.path.join('output/', 'Scores.png'))
    plt.close()