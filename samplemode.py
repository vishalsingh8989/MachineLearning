from sklearn import datasets
import matplotlib.pyplot as plt 

iris = datasets.load_iris()
data = iris.data
print(data.shape)

digits = datasets.load_digits()
digits.images.shape
im = plt.imshow(digits.images[-1]) 