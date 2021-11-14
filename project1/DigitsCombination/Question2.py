import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
import random
import glob


def BGR2RGB(img):
    return img[:, :, ::-1]


def BGR2GRAY(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def GRAY2RGB(img):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img


def knnClassification():
    # Train and Test knn Model

    filename = "digits.png"
    imgGray = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    # print(imgGray.shape)

    #### get all the digits
    IMG_SIZE = 20

    rowNum = imgGray.shape[0] / IMG_SIZE
    colNum = imgGray.shape[1] / IMG_SIZE

    rows = np.vsplit(imgGray, rowNum)

    digits = []
    for row in rows:
        rowCells = np.hsplit(row, colNum)
        for digit in rowCells:
            digits.append(digit)

    # convert list to np.array
    digits = np.array(digits)
    # print("digits", digits.shape)

    # labels
    DIGITS_CLASS = 10
    repeatNum = len(digits) / DIGITS_CLASS
    labels = np.repeat(np.arange(DIGITS_CLASS), repeatNum)
    # print("labels", labels.shape)

    #### get features
    features = []
    for digit in digits:
        img_pixel = np.float32(digit.flatten())
        features.append(img_pixel)

    features = np.squeeze(features)
    # print("features", features.shape)

    # shuffle features and labels
    # seed random for constant random value
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(features.shape[0])
    features, labels = features[shuffle], labels[shuffle]

    # split into training and testing
    splitRatio = [2, 1]
    sumRatio = sum(splitRatio)
    partition = np.array(splitRatio) * len(features) // sumRatio
    partition = np.cumsum(partition)

    featureTrain, featureTest = np.array_split(features, partition[:-1])
    labelTrain, labelTest = np.array_split(labels, partition[:-1])

    # print("featureTrain", featureTrain.shape)
    # print("featureTest", featureTest.shape)
    # print("labelTrain", labelTrain.shape)
    # print("labelTest", labelTest.shape)

    # Train the KNN model:
    # print("Training KNN model")
    knn = cv.ml.KNearest_create()
    knn.train(featureTrain, cv.ml.ROW_SAMPLE, labelTrain)

    # Test the created model:
    k = 4
    ret, prediction, neighbours, dist = knn.findNearest(featureTest, k)

    # Compute the accuracy:
    accuracy = (np.squeeze(prediction) == labelTest).mean() * 100
    # print("Accuracy k = {}: {}".format(k, accuracy))
    print('KNN model trained! Returning prediction and featureTest...')
    return prediction, featureTest


def combineDigits(ls):
    number_ls = ls
    number_split_ls = []

    for i in range(len(number_ls)):
        number_split = [int(i) for i in list(str(number_ls[i]))]
        number_split_ls.append(number_split)

    plt.figure(figsize=(5, 2))

    for i in range(len(number_split_ls)):
        for n in range(len(number_split_ls[i])):
            ind = np.where(prediction == number_split_ls[i][n])

            r_ind = random.choice(ind[0]).astype(int)

            ft = featureTest.reshape((-1, 20, 20))
            img = ft[r_ind]

            a, b = 1, len(number_split_ls[i])
            # imgRGB = GRAY2RGB(img)
            plt.subplot(a, b, n + 1)
            plt.imshow(img)
            plt.axis('off')

        print(number_split_ls[i])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    np.random.seed(2)
    random_ls = []
    prediction, featureTest = knnClassification()

    number = input('Input number:').strip().split()

    if number[0] == 'random':
        for i in range(0,5):
            n = random.randint(0, 999)
            random_ls.append(int(n))
            numbers_ls = random_ls

    else:
        numbers_ls = [int(i) for i in number]

    print('Your digits are:', numbers_ls)
    combineDigits(numbers_ls)


