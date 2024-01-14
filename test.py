import ctypes.wintypes
import os

import matplotlib.pyplot as plt
import numpy
import numpy as np

from ctypes import cdll


def find_files(filename, search_path):
    result = []

    # Walking top-down from the root
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result


def find_dll():
    dlls = find_files("MLCore.dll", "./")
    if (len(dlls) == 0):
        dlls = find_files("libMLCore.dll", "./")
    return dlls


# Plotting functions
# Classification

def LinearSimple():
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])

    plt.scatter(X[0, 0], X[0, 1], color='blue')
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red')
    plt.title('Linear simple')
    plt.show()
    plt.clf()


def LinearMultiple():
    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])

    plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='red')
    plt.title('Linear multiple')
    plt.show()
    plt.clf()


def XOR():
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])

    plt.scatter(X[0:2, 0], X[0:2, 1], color='blue')
    plt.scatter(X[2:4, 0], X[2:4, 1], color='red')
    plt.title('XOR')
    plt.show()
    plt.clf()


def Cross():
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 1],
                color='red')
    plt.title('Cross')
    plt.show()
    plt.clf()


def MultiLinear3Classes():
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 > p[1] and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 < p[1] and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 < p[0] - p[1] - 0.5 and p[1] < 0 else
                  [0, 0, 0] for p in X])

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    plt.title('Multi linear 3 classes')
    plt.show()
    plt.clf()


def MultiCross():
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 < abs(p[1] % 0.5) else [0, 1, 0] if abs(p[0] % 0.5) > 0.25 >= abs(p[1] % 0.5) else [0, 0, 1] for p in X])

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    plt.title('Multicross')
    plt.show()
    plt.clf()

# Regression

def LinearSimple2D():
    X = np.array([
        [1],
        [2]
    ])
    Y = np.array([
        2,
        3
    ])

    plt.scatter(X, Y)
    plt.title('Linear Simple 2D')
    plt.show()
    plt.clf()

def NonLinearSimple2D():
    X = np.array([
        [1],
        [2],
        [3]
    ])
    Y = np.array([
        2,
        3,
        2.5
    ])

    plt.scatter(X, Y)
    plt.title('Non Linear Simple 2D')
    plt.show()
    plt.clf()


def LinearSimple3D():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    Y = np.array([
        2,
        3,
        2.5
    ])

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    plt.title('Linear Simple 3D')
    plt.show()
    plt.clf()


def LinearTricky3D():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    Y = np.array([
        1,
        2,
        3
    ])

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    plt.title('Linear tricky 3D')
    plt.show()
    plt.clf()


def NonLinearSimple3D():
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    Y = np.array([
        2,
        1,
        -2,
        -1
    ])

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    plt.title('Non Linear Simple 3D')
    plt.show()
    plt.clf()


def AllGraphs():
    # Classification
    LinearSimple()
    LinearMultiple()
    XOR()
    Cross()
    MultiLinear3Classes()
    MultiCross()

    # Regression
    LinearSimple2D()
    NonLinearSimple2D()
    LinearSimple3D()
    LinearTricky3D()
    NonLinearSimple3D()


# Main

dlls = find_dll()
if len(dlls) == 0:
    print("Couldn't find dll named 'MLCore.dll' nor 'libMLCore.dll'.")
    exit()

dllPath = dlls[0]
print("Using '" + dllPath + "'.")
libc = cdll.LoadLibrary(dllPath)

# Prototypes de fonction
INT = ctypes.wintypes.INT;
REAL = ctypes.c_double;

ND_POINTER_INT = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C")
ND_POINTER_FLOAT = np.ctypeslib.ndpointer(REAL, ndim=1, flags="C")


libc.mlpCreate.argtypes = [ND_POINTER_INT, ctypes.wintypes.INT]
libc.mlpCreate.restype = np.int32

#	ML_API void mlpPropagate(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification);
#libc.mlpPropagate.argtypes = [INT,ND_POINTER_FLOAT,INT,ctypes.c_bool]

#	ML_API void mlpTrain(TypeId id, const Real* rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight, const Real* rawExcpectedOutputs, Integer rawExcpectedOutputsWidth, Integer rawExcpectedOutputsHeight, bool isClassification = true, float alpha = 0.01f, Integer maxIter = 1000);
libc.mlpTrain.argtypes = [INT,ND_POINTER_FLOAT,INT,INT,ND_POINTER_FLOAT,INT,INT,ctypes.c_bool,REAL,INT]


libc.mlpPredict.argtypes = [ctypes.wintypes.INT, ND_POINTER_FLOAT, ctypes.wintypes.INT, ctypes.c_bool]
libc.mlpPredict.restype = REAL



width_size = 300
height_size = 300
width_points = 100
height_points = 100
entries = np.array([2,2,1], np.int32)
count = 3

# Init lib
libc.initialize()
id = libc.mlpCreate(entries, count)

# Dataset
X = np.array([
    [1, 1],
    [2, 3],
    [3, 3]
],np.float64)
Y = np.array([
    1,
    -1,
    -1
],np.float64)

#	ML_API void mlpTrain(TypeId id, const Real* rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight, const Real* rawExcpectedOutputs, Integer rawExcpectedOutputsWidth, Integer rawExcpectedOutputsHeight, bool isClassification = true, float alpha = 0.01f, Integer maxIter = 1000);

libc.mlpTrain(id, X.ravel(), X[0].size, X.size, Y.ravel(), Y[0].size, Y.size, True, 0.1, 10)

# Affichage points
test_X = np.array([[w / width_points, h / height_points] for w in range(0,width_size) for h in range(0, height_size)], np.float64)
test_colors = ['lightblue' if libc.mlpPredict(id, input_x, width_points*height_points, False) >= 0 else 'pink' for input_x in test_X]

plt.scatter(test_X[:, 0], test_X[:, 1], c=test_colors)

# Affichage points dataset
plt.scatter(X[0, 0], X[0, 1], color='blue')
plt.scatter(X[1:3, 0], X[1:3, 1], color='red')
plt.title('Linear simple')
plt.show()
plt.clf()

#LinearSimple2D()

libc.mlpDelete(id)
libc.shutdown()