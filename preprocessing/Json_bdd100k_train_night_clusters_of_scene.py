# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:21:19 2020

@author: JimmyChen
"""

import json
import os

path = "./bdd100k/Labels/100k/train"
# files: list of Json files for labels (dataset bdd100k)
files = os.listdir(path)
list_train_filepath_dict = {"city street":[],
                            "highway":[],
                            "residential":[]}
# define the path to the text file to which the name of each image file is written
train_imagepath_txtpath_dict = {"city street":"./" + "train_night_clear_" + "city_street" + ".txt",
                                "highway":"./" + "train_night_clear_" + "highway" + ".txt",
                                "residential":"./" + "train_night_clear_" + "residential" + ".txt"}

# list of scene
list_scene_name = ['city street',
                   'highway',
                   'residential']

list_object_name = ['car',
                    'traffic sign',
                    'traffic light',
                    'person',
                    'truck']


#Image Resolution
picture_width = 1280
picture_height = 720

# go over all the Json files
file_count = 0
for file in files:
    file_path = path + "/" + file
    # append all the paths to Json files to this list
    #list_train_filepath.append(file_path)
    # Open the Json file
    with open(file_path,'r',encoding = 'utf-8') as load_f:
        # Load the data from Json file
        load_dict = json.load(load_f)
        # define the exact filename of labels of each image

        if load_dict['attributes']['weather']=="clear" and  load_dict['attributes']['timeofday']=="night":
            file_count += 1

            classes_indicator = False
            for object_dict in load_dict["frames"][0]["objects"]:
                if object_dict["category"] in list_object_name:
                    classes_indicator = True
                    break
            ####################################
            if classes_indicator == True:

                if file_count <= 19655 and load_dict['attributes']['scene'] in list_scene_name:
                    list_train_filepath_dict[load_dict['attributes']['scene']].append(file_path)

# write the path to each image to "train_imagepath_txtpath"

for scene in list_scene_name:
    file_imagepath = open(train_imagepath_txtpath_dict[scene],'w')
    for image_path in list_train_filepath_dict[scene]:
        image_path = 'data/custom/images/train/night_clear/' + image_path[-22:-5] + '.jpg'
        file_imagepath.write(image_path+'\n')
    file_imagepath.close()
