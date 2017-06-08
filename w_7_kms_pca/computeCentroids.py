import numpy as np

def computeCentroids(X, idx, K):
    #COMPUTECENTROIDS returs the new centroids by computing the means of the
    #data points assigned to each centroid.
    #   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
    #   computing the means of the data points assigned to each centroid. It is
    #   given a dataset X where each row is a single data point, a vector
    #   idx of centroid assignments (i.e. each entry in range [1..K]) for each
    #   example, and K, the number of centroids. You should return a matrix
    #   centroids, where each row of centroids is the mean of the data points
    #   assigned to it.
    #

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))


    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids(i, :)
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    # Note: You can use a for-loop over the centroids to compute this.
    #

    # for each centroid
    for j in xrange(K):

        # find training example indices that are assigned to current centroid
        # notice the [0] indexing - it's necessary because of np.nonzero()'s
        #   two-array output
        # print j
        centroid_examples = np.nonzero(idx == j)[0]
        # print "-------------------centroid---------------------"
        # print centroid_examples

        # compute mean over all such training examples and reassign centroid
        centroids[j,:] = np.mean( X[centroid_examples,:], axis=0 )


    # =============================================================

    return centroids

