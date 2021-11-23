'''
source: https://github.com/wkentaro/labelme/issues/780
source code: https://github.com/JasgTrilla/labelme/blob/master/labelme/cli/folder_to_dataset.py

HOW TO USE:
its named "folder_to_dataset.py", it converts not a single json file into dataset, but a whole folder of json files into a ready to training dataset.

OUTPUT: as output it drops the folders "training/images" and "training/labels" with the png files obtained from his correspondent json file, the png files are named as a sequence of numbers started by default from "1"

PARAMETERS: folder_to_dataset.py receives as input the folder which contains the json files you want to convert into dataset, also has an optional parameter named "-startsWith" which sets the first number to start the sequence of png output files.

Example:
the command: "folder_to_dataset.py myJsonsFolderPath" will drop: 1.png, 2.png, 3.png, … in training/images and training/labels folders
the command: "folder_to_dataset.py myJsonsFolderPath -startsWith 5“ will drop: 5.png, 6.png, 7.png, … in training/images and training/labels folders

script features:
*shows dataset building progress by percents
*skip no Json files without interrupt the process
*allows dataset updating by “startsWith” parameter
-the picture “example” shows how the script works
'''

import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils

def main():
    
    logger.info("This script will handle multiple JSON files from a folder to generate a real-use dataset.")

    parser = argparse.ArgumentParser()
    parser.add_argument("jsons_folder")
    parser.add_argument("-startsWith", default=None)
    args = parser.parse_args()
    jsons_folder = args.jsons_folder

    #define the start name(int) for the output files (output files will be named sequently starting from this number)
    if args.startsWith is None:
    	startsWith=0
    else:
    	startsWith=int(args.startsWith)-1

	#create the folders in which the output files will be saved
    out_dir = "training/images"
    out_dir2 = "training/labels"
    if not osp.exists("training"):
        os.mkdir("training")
        os.mkdir(out_dir)
        os.mkdir(out_dir2)
    else:
        logger.warning("the 'training' folder already exists, files in it with the same names as new ones, will be overwritten. (to avoid overwriting, please use the parameter '-startsWith' )")
        if(input("do you want to continue? y/n: ")=="n"):
          exit()
        if not osp.exists(out_dir):
            os.mkdir(out_dir)
        if not osp.exists(out_dir2):
            os.mkdir(out_dir2)

    #convert folder of .json files into dataset
    path= jsons_folder
    dataset= os.listdir(path)
    datasetSize=len(dataset)
    filesSkipped=[]
    print("Loaded folder:",jsons_folder)
    print("Building dataset...")
    for y,jsonFileName in enumerate(dataset):
        jsonFileType=jsonFileName.split(".")[1]
        #skip noJson files
        if(jsonFileType!="json"):
        	logger.warning("'"+jsonFileType+"'"+ " files cannot be converted (file skipped)")
        	filesSkipped.append(jsonFileName)
        else:
	        y+=1
	        outputFileName=str(y+startsWith-len(filesSkipped))
	        jsonFilePath=path+"/"+jsonFileName
	        data = json.load(open(jsonFilePath))
	        imageData = data.get("imageData")

	        if not imageData:
	            imagePath = os.path.join(os.path.dirname(jsons_folder), data["imagePath"])
	            with open(imagePath, "rb") as f:
	                imageData = f.read()
	                imageData = base64.b64encode(imageData).decode("utf-8")
	        img = utils.img_b64_to_arr(imageData)

	        label_name_to_value = {"_background_": 0}
	        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
	            label_name = shape["label"]
	            if label_name in label_name_to_value:
	                label_value = label_name_to_value[label_name]
	            else:
	                label_value = len(label_name_to_value)
	                label_name_to_value[label_name] = label_value
	        lbl, _ = utils.shapes_to_label(
	            img.shape, data["shapes"], label_name_to_value
	        )

	        label_names = [None] * (max(label_name_to_value.values()) + 1)
	        for name, value in label_name_to_value.items():
	            label_names[value] = name

	        lbl_viz = imgviz.label2rgb(
	            label=lbl, image=imgviz.asgray(img), label_names=label_names, loc="rb"
	        )

	        #Save the output files(images and labels) in the created folder above
	        PIL.Image.fromarray(img).save(osp.join(out_dir, outputFileName+".png"))
	        utils.lblsave(osp.join(out_dir2, outputFileName+".png"), lbl)
	        print(round(((y*100)/datasetSize),1),"%")
	        #PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "noImporta.png"))

	        #with open(osp.join(out_dir, "label_names.txt"), "w") as f:
	        #    for lbl_name in label_names:
	        #        f.write(lbl_name + "\n")

    logger.info(str(y-len(filesSkipped))+" image/label matches was created in folder 'training'")
    logger.warning(str(len(filesSkipped))+" Skipped files:")
    for z in filesSkipped:
    	print("*",z,"(skipped)") 


if __name__ == "__main__":
    main()