import tensorflow as tf
import numpy as np

x = []
y = []
num_data = 0
vector_dim = 52739  # numer of different items in recsys15
next_index = 0  # the next index used by next_batch()
num_examples = 23000000


def next_batch(batch_size):
    if len(x) == 0:
        __load_session_data()
    if next_index+batch_size+1 > num_data:
        diff = num_data - (next_index+batch_size+1)
        print('Not enough data remaining to create next batch, '+diff+' values missing')
        return -1, -1
    # transform to one-Hot-vector
    nbatch_x = x[next_index:next_index+batch_size]
    nbatch_y = y[next_index:next_index+batch_size]

    # thx to https://stackoverflow.com/a/29831596
    b = np.zeros((len(nbatch_x), vector_dim))
    b[np.arange(len(nbatch_x)), nbatch_x] = 1
    nbatch_x = b

    c = np.zeros((len(nbatch_y), vector_dim))
    c[np.arange(len(nbatch_y)), nbatch_y] = 1
    nbatch_y = c
    # feed dict does not support tensors somehow...
    # nbatch_x = tf.one_hot(nbatch_x, vector_dim)
    # nbatch_y = tf.one_hot(nbatch_y, vector_dim)
    # update next pointer
    global next_index
    next_index = next_index + batch_size
    return nbatch_x, nbatch_y


def __load_session_data(infile='../../ANN_DATA/RecSys15/clicks_item_to_item.txt'):
    print('Loading data from '+infile)
    line = ""
    with open(infile, 'rt') as read:
        i = 1
        while line is not None:
            line = read.readline()
            split = line.split(',')
            if len(split) < 2:
                print('format err at line '+str(i)+'. Line was ('+line+')')
                break
            x.append(int(split[0]))
            y.append(int(split[1]))
    global num_data
    num_data = len(x)
