import math
import numpy as np

def loss1():
    softmax_output = [0.7, 0.1, 0.2]

    target_output = [1, 0, 0]

    loss = -(math.log(softmax_output[0]) * target_output[0] +
            math.log(softmax_output[1]) * target_output[1] +
            math.log(softmax_output[2]) * target_output[2])

    print(loss)

def loss2():
    softmax_output = [0.7, 0.1, 0.2]

    target_output = [1, 0, 0]

    loss = -(math.log(softmax_output[0]) * target_output[0])

    print(loss)

def batch_sem_numpy():
    softmax_outputs = [[0.7, 0.1, 0.2],
                       [0.1, 0.5, 0.4],
                       [0.02, 0.9, 0.08]]

    class_targets = [0, 1, 1]

    for targ_idx, distribution in zip(class_targets, softmax_outputs):
        print(distribution[targ_idx])

def batch_com_numpy():
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])

    class_targets = [0, 1, 1]

    print(softmax_outputs[[0, 1, 2], class_targets])

def loss_batch_com_numpy():
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])

    class_targets = [0, 1, 1]

    print(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))


def media_loss_batch_com_numpy():
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])

    class_targets = [0, 1, 1]

    neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
    avarege_loss = np.mean(neg_log)
    print(avarege_loss)

def media_loss_multidimensional_targets():
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])

    class_targets = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]])

    if len(class_targets.shape) == 1:
        correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
    elif len(class_targets.shape) == 2:
        correct_confidences = np.sum(softmax_outputs * class_targets, axis = 1)

    neg_log = -np.log(correct_confidences)

    avarage_loss = np.mean(neg_log)
    print(avarage_loss)

if __name__ == '__main__':
    loss_batch_com_numpy()
    media_loss_batch_com_numpy()
    media_loss_multidimensional_targets()