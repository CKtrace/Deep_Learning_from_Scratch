import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x, type(x))

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

print(x+y)
print("-"*10)
print(x-y)
print("-"*10)
print(x*y)
print("-"*10)
print(x/y)
print("-"*10)
print(x / 2.0)

A = np.array([[1, 2], [3, 4]])
print(A.shape, A.dtype)

B = np.array([[3, 0], [0, 6]])

print(A+B)
print("-"*10)
print(A*B)
print("-"*10)
print(A)
print("-"*10)
print(A*10)
print("-"*10)
print(A.dtype)

X = np.array([[51, 55], [14, 19], [0, 4]])

for row in X:
    print(row)
    
print("-"*10)
    
X = X.flatten()
print(X)

X[np.array([0, 2, 4])]
print(X>15)
print("-"*10)
print(X[X>15])