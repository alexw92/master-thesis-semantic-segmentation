from tensorflow import nn
import tensorflow as tf


def recall(predict, y, k=20, sol_size=1):
    """
    Returns the recall value the prediction. The correct value has to be in the
    top k elements of the predicted vector
    :param predict: predicted values from the nn 1-Hot notation
    :param y: correct values 1-Hot notation
    :param k: top k values are selected, default=20
    :param sol_size: the number of items of the correct solution to be selected, default=1
    :return: the calculated recall value
    """
    p_vals, p_ind = nn.top_k(predict, k)  # take top 20 elements of prediction
    y_vals, y_ind = nn.top_k(y, sol_size)  # take the solution
    with tf.Session() as sess:
        p_ind, y_ind = sess.run([p_ind, y_ind])
    retrieved_items = set(p_ind)
    relevant_items = set(y_ind)
    intersect = relevant_items.intersection(retrieved_items)
    recall_val = len(intersect)/len(relevant_items)
    return recall_val


def mrr(predict):
    None
