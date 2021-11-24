from dataset_core import create_dataset
import matplotlib.pyplot as plt
import numpy as np
import labelme2coco

srcPaths = ['rawdata/screenshot/dataset/ss1', 'rawdata/screenshot/dataset/ss2', 'rawdata/screenshot/dataset/ss3', 'rawdata/screenshot/dataset/ss4']
datasetfilename = 'cvdataset.npz'

if create_dataset(datasetfilename, srcPaths):

    data = np.load(datasetfilename, allow_pickle=True)
    imgList = data['images']
    labelList = data['labels']
    
    print(f'{imgList.shape = }')
    print(f'{labelList.shape = }')
    
    # Display 3rd image and label
    i = -1
    img = imgList[i]
    label = labelList[i]
    print(labelList)
    
    # Convert BGR to RGB

    imgRGB = img[:, :, ::-1]

    # Show image
    plt.imshow(imgRGB)  
    plt.title(label)
    plt.axis('off')
    plt.show()
