import labelme
import os, sys

path = "..\\screenshot\\"
dirs = os.listdir(path)
i = 1

for file in dirs:
    if file.endswith(".json"):
        if os.path.isfile(path + file):
            my_dest = "ss" + str(i)
            os.system("mkdir " + my_dest)
            os.system("labelme_json_to_dataset " + file + " -o " + my_dest)
            i += 1
