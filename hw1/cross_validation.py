import argparse
import random
from collections import namedtuple
import numpy as np
from knn import Knearest, Numbers
 
random.seed(20170830)


# READ THIS FIRST
# In n-fold cross validation, all the instances are split into n folds
# of equal sizes. We are going to run train/test n times.
# Each time, we use one fold as the testing test and train a classifier
# from the remaining n-1 folds.
# In this homework, we are going to split based on the indices
# for each instance.

# SplitIndices stores the indices for each train/test run,
# Indices for the training set and the testing set 
# are respectively in two lists named 
# `train` and `test`.

SplitIndices = namedtuple("SplitIndices", ["train", "test"])

def split_cv(length, num_folds):
    """
    This function splits index [0, length - 1) into num_folds (train, test) tuples.
    """
    splits = [SplitIndices([], []) for _ in range(num_folds)]
    
    
    indices = list(range(length))
    random.shuffle(indices)
    count=1
    fold=[0,0,0,0,0]
    for i in range(0,5):
        fold[i]=list()
    #splitting indices to training and testing
    for i in indices:
        if (count == 1):
            fold[0].append(i)
            count+=1
        elif(count ==2):
            fold[1].append(i)
            count+=1
        elif(count ==3):
            fold[2].append(i)
            count+=1
        elif(count ==4):
            fold[3].append(i)
            count+=1
        elif(count ==5):
            fold[4].append(i)
            count=1
    splits[0]=SplitIndices(fold[1]+fold[2]+fold[3]+fold[4],fold[0])
    splits[1]=SplitIndices(fold[0]+fold[2]+fold[3]+fold[4],fold[1])
    splits[2]=SplitIndices(fold[1]+fold[0]+fold[3]+fold[4],fold[2])
    splits[3]=SplitIndices(fold[1]+fold[2]+fold[0]+fold[4],fold[3])
    splits[4]=SplitIndices(fold[1]+fold[2]+fold[3]+fold[0],fold[4])
    # Finish this function to populate `folds`.
    # All the indices are split into num_folds folds.
    # Each fold is the testing set in a split, and the remaining indices
    # are added to the corresponding training set.
    #print (splits[0])
    return splits


def cv_performance(x, y, num_folds, k):
    """This function evaluates average accuracy in cross validation."""
    length = len(y)
    splits = split_cv(length, num_folds)
    accuracy_array = []

    for split in splits:
        # Finish this function to use the training instances 
        # indexed by `split.train` to train the classifier,
        # and then store the accuracy 
        # on the testing instances indexed by `split.test`
        
        knn=Knearest([x[a] for a in split.train],[y[a] for a in split.train],k)
        confusion = knn.confusion_matrix([x[a] for a in split.test],[y[a] for a in split.test])
        accuracy=knn.accuracy(confusion)
        accuracy_array.append(accuracy)

    return np.mean(accuracy_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()
    
    data = Numbers("../data/mnist.pkl.gz")
    x, y = data.train_x, data.train_y
    if args.limit > 0:
        x, y = x[:args.limit], y[:args.limit]
    best_k, best_accuracy = -1, 0
    for k in [1, 3, 5, 7, 9]:
        accuracy = cv_performance(x, y, 5, k)
        print("%d-nearest neighber accuracy: %f" % (k, accuracy))
        if accuracy > best_accuracy:
            best_accuracy, best_k = accuracy, k
    knn = Knearest(x, y, best_k)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy = knn.accuracy(confusion)
    print("Accuracy for chosen best k= %d: %f" % (best_k, accuracy))
    
