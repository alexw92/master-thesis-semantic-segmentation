import os
import sys
import random





def create_de_dataset(flist_de, flist_eu):
    list_eu = []
    counter = 0
    with open(flist_eu) as read:
        l = read.readlines()
        list_eu = [x.split(" ")[0] for x in l]
    list_de_lines = []
    with open(flist_de) as read:
        list_de_lines = read.readlines()
    random.shuffle(list_de_lines)
    with open('de_ex_eu_list.txt', 'wt+') as write:
        for de_line in list_de_lines:
            if de_line.split(" ")[0] in list_eu:
                print("schon vorhanden")
                counter = counter+1
            else:
              #  pass
                write.write(de_line)
    print(counter)








# python extractds.py list_de list_eu_train
if __name__ == "__main__":
    pass
    if len(sys.argv)<3:
        print("Insufficient args. Example: python extractds.py list_de list_eu_train")
        exit()
    list_de = sys.argv[1]
    list_eu = sys.argv[2]
    create_de_dataset(list_de, list_eu)