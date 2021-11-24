import os
import cv2 as cv
import matplotlib.pyplot as plt

imgFile = 'ss4.png'
img = cv.imread(imgFile)

print(f'{img.shape = }')
# numan_pt = (732, 20), (828, 130), 'numan'

if not os.path.exists('dataset'):
    os.mkdir('dataset')

folderPath = os.path.splitext(imgFile)[0]
folderPath = f'dataset/{folderPath}'

if not os.path.exists(folderPath):
    os.mkdir(folderPath)

point_ls_of_ls = []

point_ls = []
point_ls.append(((732, 20), (828, 130), 'numan'))
point_ls.append(((426, 386), (515, 497), 'goke'))
point_ls.append(((122, 12), (207, 146), 'saseendran'))
point_ls.append(((412, 30), (508, 148), 'inamul'))
point_ls.append(((188, 350), (91, 223), 'mahmuda'))
point_ls.append(((411, 331), (488, 226), 'gavin'))
point_ls.append(((718, 315), (795, 212), 'azureen'))
point_ls.append(((75, 455), (158, 360), 'jincheng'))
point_ls.append(((716, 380), (815, 505), 'afiq'))

point_ls_of_ls.append(point_ls)

point_ls2 = []
point_ls2.append(((402, 177), (897, 773), 'numan'))
point_ls2.append(((882, 7), (960, 97), 'saseendran'))
point_ls2.append(((1302, 6), (1373, 97), 'inamul'))
point_ls2.append(((239, 25), (308, 104), 'mahmuda'))
point_ls2.append(((457, 27), (523, 114), 'gavin'))
point_ls2.append(((1090, 17), (1156, 98), 'azureen'))

point_ls_of_ls.append(point_ls2)

point_ls3 = []
point_ls3.append(((100, 17), (267, 201), 'numan'))
point_ls3.append(((159, 295), (323, 487), 'inamul'))
point_ls3.append(((635, 54), (792, 258), 'mahmuda'))
point_ls3.append(((1093, 61), (1235, 242), 'gavin'))
point_ls3.append(((1562, 43), (1699, 221), 'azureen'))
point_ls3.append(((594, 273), (734, 422), 'jincheng'))
point_ls3.append(((1119, 288), (1281, 492), 'goke'))

point_ls_of_ls.append(point_ls3)

point_ls4 = []
point_ls4.append(((503, 156), (1014, 717), 'numan'))
point_ls4.append(((221, 20), (313, 113), 'inamul'))
point_ls4.append(((1285, 23), (1367, 112), 'mahmuda'))
point_ls4.append(((443, 12), (521, 100), 'azureen'))
point_ls4.append(((877, 52), (922, 99), 'jincheng'))
point_ls4.append(((655, 3), (732, 93), 'saseendran'))
point_ls4.append(((1080, 26), (1157, 109), 'afiq'))

point_ls_of_ls.append(point_ls4)

count = 1
a, b = 3, 3

# rectangle: (x1,y1), (x2,y2)
# crop: [y1:y2, x1:x2]

# for plist in point_ls_of_ls:
for v in point_ls4:
    ((x1, y1), (x2, y2), label) = v

    if y2 < y1:
        y = y2
        y2 = y1
        y1 = y

    if x2 < x1:
        x = x2
        x2 = x1
        x1 = x

    cropped_img = img[y1:y2, x1:x2].copy()
    print(cropped_img.shape)
    # cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    saveFileName = folderPath + '/' + label + '.png'
    cv.imwrite(saveFileName, cropped_img)

    # plt.subplot(a, b, count)
    # plt.imshow(cropped_img[:, :, ::-1])
    # plt.title(label)
    # count += 1

# plt.axis('off')
# plt.tight_layout()
# plt.show()
