import sys
import os, random
import shutil

current_path = "./coreless_5000/"
test_size = 1000

train_path = "./train_data/"
test_path = "./test_data/"
file_list = []

for root, dirs, files in os.walk(current_path+"Image"):
    for file_name in files:
        file_list.append(file_name.split('.')[0])

random.shuffle(file_list)

for i in range (len(file_list)):
    print(i)
    if(i < test_size):
        shutil.copy(current_path+"Image/"+file_list[i]+".jpg", test_path+"Image/"+file_list[i]+".jpg")
    else:
        shutil.copy(current_path+"Image/"+file_list[i]+".jpg", train_path+"Image/"+file_list[i]+".jpg")
        shutil.copy(current_path+"Annotation/"+file_list[i]+".txt", train_path+"Annotation/"+file_list[i]+".txt")