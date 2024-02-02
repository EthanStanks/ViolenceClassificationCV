import os
import cv2
import model as m
import prediction as p

TRAIN_MODEL = False
PREDICT_IMAGES = False
PREDICT_VIDEO = True

def create_model():
    # preprocess images
    xtrain, ytrain, xtest, ytest = m.preprocess()
    # create and compile model
    model = m.define_model(xtrain)
    # train, validate, and save model
    m.train_model(model, xtrain, ytrain, xtest, ytest)

def picture_classification(image_paths):
    # load the saved model
    model = m.load_model()

    predictions = []
    og_images = []
    # get each prediction
    for path in image_paths:
        prediction, og_image = p.make_prediction(model, path)
        predictions.append(prediction)
        og_images.append(og_image)
    # graph predictions
    p.graph_predictions(predictions, og_images)

def video_classification(video_name, isBlur):
    video_path = os.path.join('data/','video_input/',f'{video_name}.mp4')
    isViolence = False

    # load model
    model = m.load_model()

    # load video
    cap = cv2.VideoCapture(video_path)

    # create  videowriter only if we want to apply blur effect
    if isBlur:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4)) 
        size = (frame_width, frame_height)
        output_path = os.path.join('output/',f'{video_name}_blur.mp4')
        result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

    # while video is read
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            break

        # make prediction on frame
        prediction = p.make_frame_prediction(model, frame)

        # check if frame has violence
        if prediction == 1:
            isViolence = True
            # blur frame
            if isBlur: frame = cv2.blur(frame, (50, 50))
        
        # write and show the frame
        if isBlur: result.write(frame)
        cv2.imshow("Frame", frame)

        # exit process with esc
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return isViolence

if __name__ == '__main__':

    if TRAIN_MODEL:
        # creates the classifier and saves the model
        create_model()

    if PREDICT_VIDEO:
        # Predict if video has violence
        # Add a blur if violence is detected on a frame
        isViolence = video_classification('violence', True)

        if isViolence: print('Violence Detected')
        else: print('No Violence Detected')

    if PREDICT_IMAGES:
        # paths to each validation image
        image_paths = [os.path.join('data/','validate_images/','non1.jpg'), 
                 os.path.join('data/','validate_images/','non2.jpg'), 
                 os.path.join('data/','validate_images/','violence1.jpg'),
                 os.path.join('data/','validate_images/','violence2.jpg')]
        
        # predict each image and graph results
        picture_classification(image_paths)
        
