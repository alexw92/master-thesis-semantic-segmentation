import tensorflow as tf

f = open("..\..\simple-examples\data\ptb.test.txt", 'r')
i = 0;
for line in f:
    if i < 50:
        print(str(i) + ": " + line)
        i = i + 1

f.close();
