def test():
    f = open("..\..\ANN_DATA\simple-examples\data\ptb.test.txt", 'r')
    i = 0;
    for line in f:
        if i < 50:
            print(str(i) + ": " + line)
            i = i + 1

    f.close()


def main():
    f = open("..\..\ANN_DATA\RecSys15\yoochoose-buys.dat", 'r')
    i = 0;
    myset = set()
    for line in f:
        i = i + 1
        el = line.split(",")[0]
        myset.add(el)
        if i % 100000 == 0:
            print(i)
    f.close()
    return myset,i


main()




