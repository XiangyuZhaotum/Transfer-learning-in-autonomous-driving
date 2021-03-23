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
list_train_filepath = []
list_train_filename = []
list_test_filepath = []
list_test_filename = []
# define the path to the text file to which the name of each image file is written
train_imagepath_txtpath = "./" + "train_fakenight_clear.txt"
test_imagepath_txtpath = "./" + "test_fakenight_clear.txt"
# list of object, 10 in total
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

        if load_dict['attributes']['weather']=="clear" and  load_dict['attributes']['timeofday']=="daytime":
            file_count += 1

            classes_indicator = False
            for object_dict in load_dict["frames"][0]["objects"]:
                if object_dict["category"] in list_object_name:
                    classes_indicator = True
                    break

            if classes_indicator == True:
                if file_count <= 10713:
                    list_train_filepath.append(file_path)
                    list_train_filename.append(file[:-5])
                else:
                    list_test_filepath.append(file_path)
                    list_test_filename.append(file[:-5])
                
# write the path to each image to "train_imagepath_txtpath"
file_imagepath = open(train_imagepath_txtpath,'w')
for image_path in list_train_filepath:
    image_path = 'data/custom/images/train/fake_night_clear/' + image_path[-22:-5] + '.jpg'
    file_imagepath.write(image_path+'\n')
file_imagepath.close()

file_imagepath = open(test_imagepath_txtpath,'w')
for image_path in list_test_filepath:
    image_path = 'data/custom/images/test/fake_night_clear/' + image_path[-22:-5] + '.jpg'
    file_imagepath.write(image_path + '\n')
file_imagepath.close()
