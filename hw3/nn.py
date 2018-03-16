import argparse
import numpy as np
import pickle, gzip
import matplotlib.pyplot as plt

class Network:
    def __init__(self, sizes):
        self.L = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(n, 1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n, m) for (m,n) in zip(self.sizes[:-1], self.sizes[1:])]

    def g(self, z):
        """
        activation function
        """
        return sigmoid(z)

    def g_prime(self, z):
        """
        derivative of activation function
        """
        return sigmoid_prime(z)

    def forward_prop(self, a):
        """
        memory aware forward propagation for testing
        only.  back_prop implements it's own forward_prop
        """
        for (W,b) in zip(self.weights, self.biases):
            a = self.g(np.dot(W, a) + b)
        return a

    def gradC(self, a, y):
        """
        gradient of cost function
        Assumes C(a,y) = (a-y)^2/2
        """
        return (a - y)

    def SGD_train(self, train, epochs, eta, lam=0.0, verbose=True, test=None):
        """
        SGD for training parameters
        epochs is the number of epocs to run
        eta is the learning rate
        lam is the regularization parameter
        If verbose is set will print progressive accuracy updates
        If test set is provided, routine will print accuracy on test set as learning evolves
        """
        n_train = len(train)
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for kk in range(n_train):
                xk = train[perm[kk]][0]
                yk = train[perm[kk]][1]
                dWs, dbs = self.back_prop(xk, yk)
                # TODO: Add L2-regularization
                
                #self.weights = [(W*(1-(eta*lam/n_train))) - eta*(dW +((lam/n_train)*W)) for (W, dW) in zip(self.weights, dWs)]
                #temp = lam* self.compute_cost(xk,yk)
                self.weights = [(W - eta*(dW + lam*W)) for (W, dW) in zip(self.weights, dWs)]
                self.biases = [b - eta*db for (b, db) in zip(self.biases, dbs)]
            if verbose:
                if epoch==0 or (epoch + 1) % 15 == 0:
                    acc_train = self.evaluate(train)
                    if test is not None:
                        acc_test = self.evaluate(test)
                        print("Epoch {:4d}: Train {:10.5f}, Test {:10.5f}".format(epoch+1, acc_train, acc_test))
                    else:
                        print("Epoch {:4d}: Train {:10.5f}".format(epoch+1, acc_train))

    def back_prop(self, x, y):
        """
        Back propagation for derivatives of C wrt parameters
        """
        db_list = [np.zeros(b.shape) for b in self.biases]
        dW_list = [np.zeros(W.shape) for W in self.weights]

        a = x
        a_list = [a]
        z_list = [np.zeros(a.shape)] # Pad with throwaway so indices match
        
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            z_list.append(z)
            a = self.g(z)
            a_list.append(a)

        # Back propagate deltas to compute derivatives
        # TODO delta  =
        #print (self.L)
        
        last_layer = self.L - 1
        aa = a_list[last_layer]
        zz = z_list[last_layer]
        #delta = self.gradC(aa,y)
        delta= np.multiply(np.multiply((self.gradC(aa,y)),(self.g(zz))),(1-self.g(zz)))
        for ell in range(self.L-2,-1,-1):
            # TODO db_list[ell] =
            # TODO dWyy_list[ell] =
            # TODO delta =
            dloss_w = delta.dot(a_list[ell].T)
            dloss_b = delta
            #term4 = self.weights[ell].T
            #term5 = term4.dot(delta)
            #d1 = [(a*self.g_prime(b)) for a,b in zip(term5,z_list[ell])]
            delta = np.multiply(self.weights[ell].T.dot(delta),self.g_prime(z_list[ell]))
            #delta = np.asarray(d1)
            dW_list[ell] = dloss_w
            db_list[ell] = dloss_b
            pass

        return (dW_list, db_list)

    def evaluate(self, test):
        """
        Evaluate current model on labeled test data
        """
        ctr = 0
        for x, y in test:
            yhat = self.forward_prop(x)
            ctr += np.argmax(yhat) == np.argmax(y)
        return float(ctr) / float(len(test))

    def compute_cost(self, x, y):
        """
        Evaluate the cost function for a specified
        training example.
        """
        a = self.forward_prop(x)
        return 0.5*np.linalg.norm(a-y)**2

def sigmoid(z, threshold=20):
    z = np.clip(z, -threshold, threshold)
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def mnist_digit_show(flatimage, outname=None):

    import matplotlib.pyplot as plt

    image = np.reshape(flatimage, (-1,14))

    plt.matshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    if outname:
        plt.savefig(outname)
    else:
        plt.show()

if __name__ == "__main__":

    f = gzip.open('../data/tinyMNIST.pkl.gz', 'rb') # change path to ../data/tinyMNIST.pkl.gz after debugging
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train, test = u.load()

    nn = Network([196,50,10])
    nn.SGD_train(train, epochs=200, eta=0.25, lam=0.0001, verbose=True, test=test)
