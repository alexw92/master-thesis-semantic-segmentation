import tensorflow as tf
from tensorflow.contrib.training.python.training.resample import resample_at_rate


def test():
    f = open("..\ANN_DATA\simple-examples\data\ptb.test.txt", 'r')
    i = 0;
    for line in f:
        if i < 50:
            print(str(i) + ": " + line)
            i = i + 1

    f.close()


def main():
    f = open("..\ANN_DATA\RecSys15\yoochoose-buys.dat", 'r')
    i = 0;
    myset = set()
    for line in f:
        i = i + 1
        el = line.split(",")[0]
        myset.add(el)
        if i % 100000 == 0:
            print(i)
    f.close()
    return myset, i


x1 = tf.constant(5)
x2 = tf.constant(6)
# From the tensorflow 1.0.0 release notes:
# tf.mul, tf.sub and tf.neg are deprecated
# in favor of tf.multiply, tf.subtract and tf.negative.﻿
result = tf.multiply(x1, x2)
print(result)
# Variante 1:
# sesh = tf.Session()
# print(sesh.run(result))
# sesh.close()

# Better:  (automically closes session)
with tf.Session() as sesh:
    output = sesh.run(result)
    print(output)

print(output)



