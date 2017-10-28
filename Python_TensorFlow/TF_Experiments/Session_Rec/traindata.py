import tensorflow as tf

x = []
y = []
num_data = 0


def next_batch(batch_size):
    if len(x) == 0:
        __load_session_data()

    # transform to one-Hot-vector


def __load_session_data(infile='../../ANN_DATA/RecSys15/clicks_item_to_item.txt'):
    print('Loading data from '+infile)
    line = ""
    with open(infile, 'rt') as read:
        while line is not None:
            line = read.readline()
            split = line[',']
            if len(split) < 2:
                print('format err at line='+line)
            x.append(int(split[0]))
            y.append(int(split[1]))
    global num_data
    num_data = len(x)


__load_session_data()


