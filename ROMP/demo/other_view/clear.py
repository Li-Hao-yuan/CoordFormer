import os

root = "/home/lihaoyuan/vomp/ROMP/demo/other_view/"
file_name_list = ["back/","up/","left/","right/"]

for file_name in file_name_list:
    print("clear ",file_name)
    for file in os.listdir(root+file_name):
        file_path = root+file_name+file
        os.remove(file_path)