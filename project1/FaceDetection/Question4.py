import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys


def BGR2RGB(img):
    return img[:, :, ::-1]


def BGR2GRAY(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def GRAY2RGB(img):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img


def imageRead(path):
    img = cv.imread(path)

    if img is None:
        sys.exit("Could not read the image.")

    img_gray = BGR2GRAY(img)
    # load cascade model classifier
    cascade = cv.CascadeClassifier(script_path)

    # detect faces
    faces = cascade.detectMultiScale(img_gray, 1.1, 5)

    color = (0, 255, 255)
    thick = 2
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), color, thick)

    # BGR to RGB
    print('Image shape:', img.shape)
    cv.imshow('img', img)
    cv.waitKey()
    cv.destroyAllWindows()


def videoRead(path):

    # Read video from path
    capture = cv.VideoCapture(path)

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

        gray = BGR2GRAY(frame)
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

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

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

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
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 7)
                # Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

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


if __name__ == '__main__':

    arg_name = sys.argv[1]
    print(f'Opening {arg_name}...')

    script_path = "haarcascade_frontalface_alt2.xml"
    face_cascade = cv.CascadeClassifier(script_path)

    image_ext = ['png', 'jpg']
    video_ext = ['avi', 'mp4']

    if arg_name[-3:] in image_ext:
        imageRead(arg_name)

    if arg_name[-3:] in video_ext:
        videoRead(arg_name)

    if arg_name == 'openCamera':
        openCamera()

