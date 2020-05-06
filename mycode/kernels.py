import numpy as np
from numpy import linalg
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel 


class Kernel(object):
    
    def __call__(self, x, x_i):
        raise NotImplementedError()

    def grad_x(self, x, x_i):
        raise NotImplementedError()    


class Linear(Kernel):

    def __call__(self, x, x_i):
        x = np.atleast_2d(x)
        x_i = np.atleast_2d(x_i)
        return linear_kernel(x, x_i)

    def grad_x(self, x, x_i):
        grad = x
        return 2 * grad if np.linalg.norm(x - x_i) < 1e-8 else grad
    

class Polynomial(Kernel):

    def __init__(self, p=2, c=1.0, gamma=1.0):
        self.p = p
        self.c = c
        self.gamma = gamma

    def _poly(self, x, x_i, c, p, gamma):
        x = np.atleast_2d(x)
        x_i = np.atleast_2d(x_i)
        return polynomial_kernel(x, x_i, p, gamma, c)

    def __call__(self, x, x_i):
        k = self._poly(x, x_i, self.c, self.p, self.gamma)
        return k

    def grad_x(self, x, x_i):
        grad = self.p * self.gamma * self._poly(x, x_i, self.c, self.p-1, self.gamma) * x
        return grad * 2 if np.linalg.norm(x - x_i) < 1e-8 else grad


class Gaussian(Kernel):

    def _rbf(self, x, x_i):
        x = np.atleast_2d(x)
        x_i = np.atleast_2d(x_i)
        out = rbf_kernel(x, x_i, self.gamma)

        return out

    def __init__(self, gamma=1.0):
        self.gamma = gamma
        
    def __call__(self, x, x_i):
        return self._rbf(x, x_i)

    def grad_x(self, x, x_i):
        return 2 * self.gamma * self._rbf(x, x_i) * (x - x_i)


if __name__ == "__main__":
    
    # ============== Linear Kernel Tests ===============
    k = Linear()
    print(k(np.array([[1,2],[3,4]]), np.array([[10,20],[30,40]])))
    # CArray([[  50.  110.]
    #  [ 110.  250.]])

    array = np.array([[15,25],[45,55]])
    vector = np.array([2,5])
    print(k.grad_x(array, vector))
    # CArray([[15 25]
    #      [45 55]])

    print(k.grad_x(vector, vector))
    # CArray([ 4 10])

    print("----------------")
    # ============== Polynomial Kernel Tests ============
    k = Polynomial(p=3, c=2, gamma=0.001)
    print(k(np.array([[1,2],[3,4]]), np.array([[10,20],[30,40]])))
    # CArray([[  8.615125   9.393931]
    #  [  9.393931  11.390625]])
    
    k = Polynomial(p=3, c=2, gamma=1e-4)
    array = np.array([[15,25],[45,55]])
    vector = np.array([2,5])
    print(k.grad_x(array, vector))
    # CArray([[ 0.01828008  0.0304668 ]
    #      [ 0.05598899  0.06843098]])

    print(Polynomial().grad_x(vector, vector))
    # CArray([ 240.  600.])

    print("----------------")
    # ============== Gaussian Kernel Tests ==============
    k = Gaussian(gamma=0.001)
    a = np.array([[1,2], [3,4]])
    b = np.array([[10,20], [30, 40]])
    print(k(a, b))
    # CArray([[0.66697681 0.10177406]
    #         [0.73712337 0.13199384]])
    
    array = np.array([[15,25],[45,55]])
    vector = np.array([2,5])
    grad = Gaussian(gamma=1e-4).grad_x(array, vector)
    print(grad)
    # CArray([[ 0.00245619  0.00377875]
    #        [ 0.00556703  0.00647329]])
    
    print(Gaussian().grad_x(vector, vector))
    # CArray([ 0.  0.])
    print("----------------")
