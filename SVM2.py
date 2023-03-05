import numpy as np
import matplotlib.pyplot as plt

def generate_data(dim, num):
    x = np.random.normal(0, 10, [num, dim])
    coef = np.random.uniform(-1, 1, [dim, 1])
    pred = np.dot(x, coef)
    pred_n = (pred - np.mean(pred)) / np.sqrt(np.var(pred))
    label = np.sign(pred_n)
    mislabel_value = np.random.uniform(0, 1, num)
    mislabel = 0
    for i in range(num):
        if np.abs(pred_n[i]) < 1 and mislabel_value[i] > 0.9 + 0.1 * np.abs(pred_n[i]):
            label[i] *= -1
            mislabel += 1
    return x, label, mislabel/num

X, y, mr = generate_data(5,1000)
X_train = X[0:800]
X_test = X[800:1000]
y_train = y[0:800]
y_test = y[800:1000]

# np.random.seed(12)
# num_observations = 50

# x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
# x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

# X = np.vstack((x1, x2)).astype(np.float32)
# y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

# y = np.where(y <= 0, -1, 1)

# plt.figure(figsize=(12,8))
# plt.scatter(X[:, 0], X[:, 1], c = y, alpha = .4)

# def Lagrangian(w, alpha):
#     first_part = np.sum(alpha)
#     second_part = np.sum(np.dot(alpha*alpha*y*y*X.T, X))
#     res = first_part - 0.5 * second_part
#     return res



class SVM2:
    def __init__(self, dim, circle = 2000, lr = 0.0001):
        """
        You can add some other parameters, which I think is not necessary
        """
        self.w = np.random.random(dim)
        self.b = 0
        self.lr = lr
        self.circle = circle

    def fit(self, X, y):
        """
        Fit the coefficients via your methods
        """
        for i in range(self.circle):
            for idx, x_i in enumerate(X):
                y_i = y[idx]
                if (y_i * (np.dot(x_i, self.w) - self.b) >= 1):
                    self.w -= self.lr * 2 * self.w
                else:
                    self.w -= self.lr * (2 * self.w - np.dot(x_i, y_i[0]))
                    self.b -= self.lr * y_i
        
    def predict(self, X, y):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        pred = np.dot(X, self.w) - self.b
        pred = np.sign(pred)
        num = y.shape[0]
        cor_num = 0
        for i in range(num):
            if(y[i] == pred[i]):
                cor_num += 1
        print(cor_num / num)

svm = SVM2(dim = 5)
svm.fit(X = X_train, y = y_train)
svm.predict(X = X_test, y = y_test)