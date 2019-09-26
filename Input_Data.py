import os
import shutil

path = 'ILSVRC2014_DET_train'
dirs = os.listdir(path)
root_target_folder = 'Training_data'

# for root, dirs, files in os.walk((os.path.normpath(path)), topdown=False):
#     for file in dirs:
#         for name in file:
#             if file.endswith('.jpeg'):
#                 print("Found")
#                 SourceFolder = os.path.join(root, name)
#                 shutil.copy2(SourceFolder, root_target_folder)

for root, dirs, files in os.walk(path):
    for file in files:
        path_file = os.path.join(root, file)
        shutil.copy2(path_file, 'Training_data')



#
# for file in dirs:
#     print(file)
