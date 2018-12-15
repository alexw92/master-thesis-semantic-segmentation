import os
import sys
import glob
import shutil


if __name__ == "__main__":
    fromList = sys.argv[1]
    fromFolder = sys.argv[2]
    toFolder = sys.argv[3]
    if not os.path.exists(toFolder):
        os.mkdir(toFolder)
    with open(fromList) as list:
        for l in list:
            file = l.split(' ')[0]
            print(file)
            full_file_name = os.path.join(fromFolder, file)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, toFolder)
            else:
                print("file not found "+file)

