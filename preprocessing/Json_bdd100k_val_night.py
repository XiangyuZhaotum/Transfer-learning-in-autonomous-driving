# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:21:19 2020

@author: JimmyChen
"""

import json
import os

path = "./bdd100k/Labels/100k/val"
# files: list of Json files for labels (dataset bdd100k)
files = os.listdir(path)
list_valid_filepath = []
list_valid_filename = []
# define the path to the text file to which the name of each image file is written
valid_imagepath_txtpath = "./" + "valid_night_clear.txt"
# define the path to the directory to which each images's labels (in form of text) are written
valid_label_filepath = "./labels/valid/" + "night_clear"
# list of object, 10 in total
list_object_name = ['car',
                    'traffic sign',
                    'traffic light',
                    'person',
                    'truck']
# Image Resolution
picture_width = 1280
picture_height = 720

# go over all the Json files
for file in files:
    file_path = path + "/" + file
    # append all the paths to Json files to this list
    # list_train_filepath.append(file_path)
    # Open the Json file
    with open(file_path, 'r', encoding='utf-8') as load_f:
        # Load the data from Json file
        load_dict = json.load(load_f)
        # define the exact filename of labels of each image

        if load_dict['attributes']['weather'] == "clear" and load_dict['attributes']['timeofday'] == "night":
            #########################
            classes_indicator = False
            for object_dict in load_dict["frames"][0]["objects"]:
                if object_dict["category"] in list_object_name:
                    classes_indicator = True
                    break
            ###########################
            if classes_indicator == True:
            ###########################

                list_valid_filepath.append(file_path)
                list_valid_filename.append(file[:-5])
                valid_label_txtpath = valid_label_filepath + "/" + file[:-5] + ".txt"
                # open the file defined in the previous step
                file_label = open(valid_label_txtpath, 'w')
                for object_dict in load_dict["frames"][0]["objects"]:
                    if "box2d" in object_dict.keys():
                        ###############################
                        if object_dict["category"] in list_object_name:
                        ###############################
                            # set_object_name.add(object_dict["category"])
                            box2d = object_dict["box2d"]
                            # relative position of center point and relative width and height w.r.t the original image
                            x_center = (box2d["x1"] + box2d["x2"]) / (2 * picture_width)
                            y_center = (box2d["y1"] + box2d["y2"]) / (2 * picture_height)
                            width = (box2d["x2"] - box2d["x1"]) / picture_width
                            height = (box2d["y2"] - box2d["y1"]) / picture_height
                            # get the index of the label corresponding to the predefined classes
                            file_label.write(str(list_object_name.index(object_dict["category"])) + ' ')
                            file_label.write(str(x_center) + ' ')
                            file_label.write(str(y_center) + ' ')
                            file_label.write(str(width) + ' ')
                            file_label.write(str(height) + ' ' + '\n')
                file_label.close()

# write the path to each image to "train_imagepath_txtpath"
file_imagepath = open(valid_imagepath_txtpath, 'w')
for image_path in list_valid_filepath:
    image_path = 'data/custom/images/valid/night_clear/' + image_path[-22:-5] + '.jpg'
    file_imagepath.write(image_path + '\n')
file_imagepath.close()
