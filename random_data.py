import sys
import os, random
import shutil
from config import HOME

current_path = "./coreless_5000/"
test_size = 1000

train_path = "./train_data/"
test_path = HOME + "/work/test_data/"
file_list = []

print(test_path)

ids = []
num_1 = 0
num_2 = 0

with open("test_data_id.txt", "w") as f:
    for root, dirs, files in os.walk(test_path + "Annotation"):
        for file_name in files:
            print(file_name)
            file_id = file_name.split(".")[0]
            if file_id == "":
                continue
            f.write(file_id)
            f.write("\n")


# for root, dirs, files in os.walk(train_path + "Annotation"):
#     for file_name in files:
#         file_id = file_name.split(".")[0]
#         if file_id == "":
#             continue
#         with open(train_path + "Annotation/" + file_name, encoding="utf-8", mode="r") as f:
#             useful = False
#             for line in f.readlines():
#                 if line.find(u" 带电芯") != -1:
#                     num_1 = num_1 + 1
#                     useful = True
#                 if line.find(u" 不带电芯") != -1:
#                     num_2 = num_2 + 1
#                     useful = True
#             if useful == False:
#                 ids.append(file_id)
# print(num_1)# 866
# print("=============\n")
# print(num_2) # 4297  ~ 1:5

# for id in ids:
#     os.remove(train_path + "Annotation/" + id +".txt")
#     os.remove(train_path + "Image/" + id +".jpg")
        
                

# for root, dirs, files in os.walk(train_path + "Annotation"):
#     for file_name in files:
#         num = 0
#         file_id = file_name.split(".")[0]
#         if file_id == "":
#             continue
#         line1 = ""
#         with open(train_path + "Annotation/" + file_name, encoding="utf-8", mode="r") as f:
#             for line in f.readlines():
#                 if num == 0:
#                     line1 = line
#                 num = num + 1
#                 if num > 1:
#                     vec_id = file_id + ("_vec%d" %num)
#                     with open(train_path + "Annotation/" + vec_id + ".txt", encoding="utf-8", mode="w") as f2:
#                         f2.write(line)
#                         f2.close
#                         shutil.copy(
#                             train_path + "Image/" + file_id + ".jpg",
#                             train_path + "Image/" + vec_id + ".jpg",
#                         )
#         if num > 1:
#             with  open(train_path + "Annotation/" + file_name, encoding="utf-8", mode="w+") as f:
#                 f.write(line1)        
#                 f.close
#             print(file_name)
#             print("\n")

# random.shuffle(file_list)

# for i in range(len(file_list)):
#     print(i)
#     if i < test_size:
#         shutil.copy(
#             current_path + "Image/" + file_list[i] + ".jpg",
#             test_path + "Image/" + file_list[i] + ".jpg",
#         )
#         shutil.copy(
#             current_path + "Annotation/" + file_list[i] + ".txt",
#             test_path + "Annotation/" + file_list[i] + ".txt",
#         )
#     else:
#         shutil.copy(
#             current_path + "Image/" + file_list[i] + ".jpg",
#             train_path + "Image/" + file_list[i] + ".jpg",
#         )
#         shutil.copy(
#             current_path + "Annotation/" + file_list[i] + ".txt",
#             train_path + "Annotation/" + file_list[i] + ".txt",
#         )
        

