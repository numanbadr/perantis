import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random


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
    rand = np.random.RandomState(2)
    shuffle = rand.permutation(features.shape[0])
    features, labels = features[shuffle], labels[shuffle]

    # split into training and testing
    splitRatio = [10, 1]
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
    k = 6
    ret, prediction, neighbours, dist = knn.findNearest(featureTest, k)

    # # Compute the accuracy:
    accuracy = (np.squeeze(prediction) == labelTest).mean() * 100
    print(f'\nk = {k}, Split ratio = {splitRatio}')
    print(f"Accuracy of pre-trained KNN model = {accuracy:.2f} %\n")
    # print('KNN model trained! Returning prediction and featureTest...')
    # return prediction, featureTest

    return knn


def generateDigits(number):
    img = cv.imread('digits.png', cv.IMREAD_GRAYSCALE)
    digit_ls = []
    random_ls = []
    number_split_ls = []

    # digits.png = 100 row x 50 column of images
    # 1 image = 20 x 20 pixels
    # pixel of total images: 2000 row x 1000 column

    # Get individual digit images in a list
    for col in range(0, 2000, 20):
        for row in range(0, 1000, 20):
            digit = img[col:col + 20, row:row + 20]
            digit_ls.append(digit)

    # vconcat: 1 image pixel = 5000 row x 1 col
    digit_arr = cv.vconcat(digit_ls)
    # print(digit_arr.shape) # quite important to look
    # cv.imshow('haa', digit_arr[:])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # plt.imshow(GRAY2RGB(digit_arr[25000:25020]))
    # plt.show()
    ## Get desired number images
    # List given by user
    if number[0] == 'random':
        for i in range(0, random.randint(2, 4)):
            n = random.randint(0, 9999)
            random_ls.append(int(n))
            numbers_ls = random_ls
    else:
        numbers_ls = [int(i) for i in number]

    print('\nYour digits are:', numbers_ls)
    print()
    img_sep = []
    ls_sep = []
    num_save = []
    for i in range(len(numbers_ls)):
        number_split = [int(i) for i in list(str(numbers_ls[i]))]
        number_split_ls.append(number_split)

        # print(numbers_ls[i])

    for i in range(len(number_split_ls)):
        for n in range(len(number_split_ls[i])):

            select_digit = int(number_split_ls[i][n])
            # print('Selected digit:', select_digit)
            a = select_digit * 5000
            b = a + 5000
            select_digit_value = []

            for j in range(a, b, 20):
                select_digit_value.append(j)

            start_index = select_digit_value[np.random.randint(
                0, len(select_digit_value))]
            img = digit_arr[start_index:start_index + 20, :]

            img_sep.append(img)
        ls_sep.append(img_sep)

        # num_img = cv.hconcat(img_sep[n])
        # plt.imshow(GRAY2RGB(num_img))
        # plt.show()

        # ls_sep.append(img_sep[n])

        # print('len imgsep',len(img_sep))
        # print(len(ls_sep))

        # print('type num_img', type(num_img))
        # print('shape num_img', num_img.shape)
        # print('type img_sep', type(img_sep))

        # print('shape imsep', num_img.shape)

        # print('len', len(img_sep))
        # Train & show comparison
        if len(img_sep) == 1:
            num_img = np.array(img_sep)
            # print('haih')
            testDigits(num_img, number_split_ls[i])
        else:
            num_img = cv.hconcat(img_sep)
            testDigits(num_img, number_split_ls[i])

        plt.imshow(GRAY2RGB(num_img))
        plt.axis('off')
        plt.show()

        img_sep = []
    # cv.imwrite('digit_image.png', num_img)

    # plt.imshow(num_save)
    # plt.axis('off')
    # plt.show()


def testDigits(img, ls):
    imgGray = img

    IMG_SIZE = 20

    # Resize
    rowNum = imgGray.shape[0] / IMG_SIZE
    colNum = imgGray.shape[1] / IMG_SIZE

    rows = np.vsplit(imgGray, rowNum)  # split each row first

    digits = []
    for row in rows:
        rowCells = np.hsplit(row,
                             colNum)  # after splitting row, split each col
        for digit in rowCells:
            digits.append(digit)  # each cell rep a particular digit

    # convert list to np.array
    digits = np.array(digits)

    # labels
    # DIGITS_CLASS = 10
    # repeatNum = len(digits) / DIGITS_CLASS
    # labels = np.repeat(np.arange(DIGITS_CLASS), repeatNum)
    # labels = np.squeeze(labels)

    new_labels = np.array(ls)

    own_features = []
    for digit in digits:
        img_pixel = np.float32(digit.flatten())
        own_features.append(img_pixel)

    own_features = np.squeeze(own_features[0:len(ls)])

    k = 4
    ret, prediction, neighbours, dist = knn.findNearest(own_features, k)

    # Compute the accuracy:

    accuracy = (np.squeeze(prediction) == new_labels).mean(
    ) * 100  # Bool error kena array kan labels, index kan features

    print('Inputted: ', ls)
    print('Predicted:', prediction.flatten())
    print("Accuracy: {:.2f} %".format(accuracy))
    print()


def MainMenu():
    print('==================\nDIGITS COMBINATION\n==================')
    print('[1] Input own digit combination')
    print('[2] Generate random digit combination')
    print('[0] Exit program')


if __name__ == '__main__':
    np.random.seed(2)

    # Train the algorithm
    knn = knnClassification()

    while True:
        MainMenu()
        
        try:
            option = int(input('Pick your option: '))

            if option == 1:
                number = input('Input number:').strip().split()
                generateDigits(number)

            elif option == 2:
                number = ['random']
                generateDigits(number)

            elif option == 0:
                print('Quitting program...')
                break

        except ValueError:
            print('Invalid option. Please try again.\n')
    exit
