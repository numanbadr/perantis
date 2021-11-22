import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt


def BGR2RGB(img):
    return img[:, :, ::-1]


def BGR2GRAY(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def GRAY2RGB(img):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img


def detectHaar(img):
    script_path = "model/haarcascade_frontalface_alt2.xml"
    face_cascade = cv.CascadeClassifier(script_path)

    # load cascade model classifier
    cascade = cv.CascadeClassifier(script_path)

    # detect faces
    img_gray = BGR2GRAY(img)
    faces = cascade.detectMultiScale(img_gray, 1.1, 5)

    color = (0, 255, 255)
    thick = 2
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), color, thick)


def detectCaffe(img):
    # image dimension
    h, w = img.shape[:2]

    # load model
    model = cv.dnn.readNetFromCaffe(
        "model/deploy.prototxt",
        "model/res10_300x300_ssd_iter_140000_fp16.caffemodel")

    # preprocessing
    # image resize to 300x300 by substraction mean vlaues [104., 117., 123.]
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.],
                                False, False)

    # set blob asinput and detect face
    model.setInput(blob)
    detections = model.forward()

    faceCounter = 0
    # draw detections above limit confidence > 0.7
    for i in range(0, detections.shape[2]):
        # confidence
        confidence = detections[0, 0, i, 2]
        #
        if confidence > 0.7:
            # face counter
            faceCounter += 1
            # get coordinates of the current detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the detection and the confidence:
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text = "{:.2f}%".format(confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText(img, text, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 2)


def detectTensorflow(img):
    h, w = img.shape[:2]

    model = cv.dnn.readNetFromTensorflow("model/opencv_face_detector_uint8.pb",
                                         "model/opencv_face_detector.pbtxt")

    # preprocessing
    # image resize to 300x300 by substraction mean vlaues [104., 117., 123.]
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.],
                                False, False)

    # set blob asinput and detect face
    model.setInput(blob)
    detections = model.forward()

    faceCounter = 0
    # draw detections above limit confidence > 0.7
    for i in range(0, detections.shape[2]):
        # confidence
        confidence = detections[0, 0, i, 2]
        #
        if confidence > 0.7:
            # face counter
            faceCounter += 1
            # get coordinates of the current detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the detection and the confidence:
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text = "{:.2f}%".format(confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText(img, text, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 200, 200), 2)
            face_count = f'Faces detected: {faceCounter}'
    # cv.putText(img, face_count, (20,75), cv.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 3)


def imageRead(img, model):

    if img is None:
        sys.exit("Could not read the image.")

    img_gray = BGR2GRAY(img)
    h, w = img.shape[:2]

    # load model
    model

    # show
    # print('Image shape:', img.shape)
    cv.imshow('img', img)
    cv.waitKey()
    cv.destroyAllWindows()


def videoRead(capture):

    # Check if connected
    if capture.isOpened() is False:
        print("Error opening camera 0")
        exit()

    # Get and print video information
    frame_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
    frame_delay = frame_count / fps

    print(f'Frame per seconds: {fps:.2f} fps')
    print(f'Video W x H: {frame_width} x {frame_height} px')
    print(f'Frame count: {frame_count}')
    print(f'Frame delay: {frame_delay:.2f} ms')

    while capture.isOpened():
        # Capture frames, if read correctly ret is True
        ret, frame = capture.read()

        # If no frame is returned, stop the loop
        if not ret:
            print("Frame stopped.")
            break

        # Load model
        # detectHaar(frame)
        # detectCaffe(frame)
        detectTensorflow(frame)

        # Show video
        cv.imshow("Camera frame", frame)

        # Set frame delay
        k = cv.waitKey(1)

        # Check if key is q then exit
        if k == ord("q"):
            break

    # Close the video
    capture.release()
    cv.destroyAllWindows()


def openCamera():
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)

    # check if connected
    if capture.isOpened() is False:
        print("Error opening camera 0")
        exit()

    print('Type s to save image\nType v to record video\nType q to exit')

    while capture.isOpened():
        # capture frames, if read correctly ret is True
        ret, frame = capture.read()

        if not ret:
            print("Didn't receive frame. Stop ")
            break

        # load model
        # detectHaar(frame)
        # detectCaffe(frame)
        detectTensorflow(frame)

        cv.imshow("Camera frame", frame)
        k = cv.waitKey(1)

        # display frame
        if k == ord("v"):
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            video_out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    print("Didn't receive frame. Stop ")
                    break

                # load model
                # detectHaar(frame)
                # detectCaffe(frame)
                detectTensorflow(frame)

                video_out.write(frame)
                cv.imshow("Camera frame", frame)

                k = cv.waitKey(1)
                # check if key is q then exit
                if k == ord("q"):
                    video_out.release()
                    cv.destroyAllWindows()
                    break

        # check if key is s then save frame
        if k == ord("s"):
            # save color frame
            cv.imwrite('frame.png', frame)

        # check if key is q then exit
        if k == ord("q"):
            capture.release()
            cv.destroyAllWindows()
            break

    capture.release()
    cv.destroyAllWindows()


def MainMenu():
    print('\n==================\nFACE DETECTION\n==================')
    print('[1] Upload an image')
    print('[2] Upload a video')
    print('[3] Open live camera')
    print('[0] Exit program')


if __name__ == '__main__':

    image_ext = ['png', 'jpg', 'jpeg']
    video_ext = ['avi', 'mp4']

    while True:
        MainMenu()

        try:
            option = int(input('Pick your option: '))

            if option == 3:
                openCamera()

            elif option == 1 or option == 2:
                file_path = str(input('Enter file path: '))

                if file_path[-3:] in image_ext:
                    img = cv.imread(file_path)

                    # detectHaar(img)
                    # detectCaffe(img)
                    model = detectTensorflow(img)
                    imageRead(img, model)

                if file_path[-3:] in video_ext:
                    capture = cv.VideoCapture(file_path)
                    videoRead(capture)
            elif option == 0:
                print('Qutting program...')
                break
        except ValueError:
            print('Invalid option. Please try again.\n')

    exit


