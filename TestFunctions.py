import matplotlib.pyplot as plt
import numpy as np
nbIteration=250
def Predict(libc, useMLP : bool, isClassification : bool, entries, X, Y, width_size : int, height_size : int, resolution : int, width_offset = 0, height_offset = 0):

    if isClassification :
        test_X = np.array([[(w / resolution) * width_size + width_offset, (h / resolution) * height_size + height_offset] for w in range(0, resolution) for h in range(0, resolution)], np.float64)
    else :
        test_X = np.array([[(w / resolution) * width_size + width_offset] for w in range(0, resolution)],np.float64)
        test_Y= []
    test_colors = []

    if useMLP:
        idMLP = libc.mlpCreate(entries, entries.size)
        #void Train(const Real* rawAllInputs, Integer inputSize, Integer inputsCount, const Real* rawExcpectedOutputs, Integer outputSize, Integer outpuCount, bool isClassification = true, Real alpha = 0.01f, Integer maxIter = 1000);
        libc.mlpTrain(idMLP, X.ravel(), np.shape(X)[1], np.shape(X)[0], Y.ravel(),  1,np.shape(Y)[0], isClassification, 0.1, nbIteration)
        for input_x in test_X:
            raveled=input_x.ravel();
            predictCount = libc.mlpPredict(idMLP,raveled , raveled.size, isClassification)
            f=libc.mlpGetPredictData(idMLP, predictCount)
            if isClassification:
                test_colors.append('lightblue' if f >= 0 else 'pink')
            else:
                print(f)
                test_Y.append(f)
                test_colors.append([1.0,1.0,1.0])
        libc.mlpDelete(idMLP)

    else:
        idLinear = libc.linearCreate(isClassification,0.01, np.shape(X)[1])
        libc.linearTrain(idLinear, nbIteration,X.ravel(), Y.ravel(), np.shape(X)[0])
        test_colors = ['lightblue' if libc.linearEvaluate(idLinear, input_x) >= 0 else 'pink' for input_x in test_X]
        libc.linearDelete(idLinear)

    # Show prediction
    if isClassification:
        plt.scatter(test_X[:, 0], test_X[:, 1], c=test_colors)
    else:
        plt.scatter(test_X, test_Y, c=test_colors)

def LinearSimple(libc, useMLP, width_size=4, height_size=4, resolution=100):

    # Dataset
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], np.float64)
    Y = np.array([
        1,
        -1,
        -1
    ], np.float64)

    entries = np.array([2, 1], np.int32)
    Predict(libc, useMLP, True, entries, X, Y, width_size, height_size, resolution)

    # Show dataset
    plt.scatter(X[0, 0], X[0, 1], color='blue')
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red')
    Show('Linear simple')

# Note: Pas très bien zoomé
def LinearMultiple(libc, useMLP, width_size=2, height_size=2, resolution=100, width_offset = 1, height_offset = 1):
    X = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    X.astype(np.float64)
    Y.astype(np.float64)

    entries = np.array([2, 1], np.int32)
    Predict(libc, useMLP, True, entries, X, Y, width_size, height_size, resolution, width_offset, height_offset)

    plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='red')
    Show('Linear multiple')


def XOR(libc, useMLP, width_size=1, height_size=1, width_points=100, resolution=100):
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], np.float64)
    Y = np.array([1, 1, -1, -1], np.float64)

    entries = np.array([2, 2, 1], np.int32)
    Predict(libc, useMLP, True, entries, X, Y, width_size, height_size,  resolution)

    plt.scatter(X[0:2, 0], X[0:2, 1], color='blue')
    plt.scatter(X[2:4, 0], X[2:4, 1], color='red')
    Show('XOR')


def Cross(libc, useMLP, width_size=2, height_size=2, resolution=100, width_offset = -1, height_offset = -1):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X], np.float64)

    X.astype(np.float64)

    entries = np.array([2, 4, 1], np.int32)
    Predict(libc, useMLP, True, entries, X, Y, width_size, height_size, resolution, width_offset, height_offset)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 1],
                color='red')
    Show('Cross')


def MultiLinear3Classes(libc, useMLP, width_size=2, height_size=2, resolution=100, width_offset = -1, height_offset = -1):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 > p[1] and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 < p[1] and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 < p[0] - p[1] - 0.5 and p[1] < 0 else
                  [0, 0, 0] for p in X], np.float64)

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

    X.astype(np.float64)
    Y.astype(np.float64)

    entries = np.array([2, 3], np.int32)
    Predict(libc, useMLP, True, entries, X, Y, width_size, height_size, resolution, width_offset, height_offset)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    Show('Multi linear 3 classes')


def MultiCross(libc, useMLP, width_size=2, height_size=2, resolution=100, width_offset = -1, height_offset = -1):
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 < abs(p[1] % 0.5) else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 >= abs(p[1] % 0.5) else [0, 0, 1] for p in X], np.float64)

    X.astype(np.float64)

    # Note: L'exemple stipule MLP (2, ?, ?, 3)... good luck
    entries = np.array([2, 3, 3, 3], np.int32)
    Predict(libc, useMLP, True, entries, X, Y, width_size, height_size, resolution, width_offset, height_offset)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    Show('Multicross')


# Regression

def LinearSimple2D(libc, useMLP, width_size=2, height_size=3, resolution=100):
    X = np.array([
        [1],
        [2]
    ], np.float64)
    Y = np.array([
        2,
        3
    ], np.float64)

    entries = np.array([1, 1], np.int32)
    Predict(libc, useMLP, False, entries, X, Y, width_size, height_size, resolution)

    plt.scatter(X, Y)
    Show('Linear Simple 2D')


def NonLinearSimple2D(libc, useMLP, width_size=3, height_size=3, resolution=100):
    X = np.array([
        [1],
        [2],
        [3]
    ], np.float64)
    Y = np.array([
        2,
        3,
        2.5
    ], np.float64)

    # Note: L'exemple stipule MLP (1, ?, 1)... good luck
    entries = np.array([1, 2, 1], np.int32)
    Predict(libc, useMLP, False, entries, X, Y, width_size, height_size, resolution)

    plt.scatter(X, Y)
    Show('Non Linear Simple 2D')

# Note: les graphs 3d sont broken (Montre un graph 2d + un graph 3d
def LinearSimple3D(libc, useMLP, width_size=3, height_size=3, resolution=100):
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ], np.float64)
    Y = np.array([
        2,
        3,
        2.5
    ], np.float64)

    entries = np.array([2, 1], np.int32)
    Predict(libc, useMLP, False, entries, X, Y, width_size, height_size, resolution)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    Show('Linear Simple 3D')


def LinearTricky3D():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ], np.float64)
    Y = np.array([
        1,
        2,
        3
    ], np.float64)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    Show('Linear tricky 3D')


def NonLinearSimple3D():
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ], np.float64)
    Y = np.array([
        2,
        1,
        -2,
        -1
    ], np.float64)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    Show('Non Linear Simple 3D')


def AllGraphs(libc, useMLP):
    # Classification
    LinearSimple(libc, useMLP)
    LinearMultiple(libc, useMLP)
    XOR(libc, useMLP)
    Cross(libc, useMLP)
    MultiLinear3Classes(libc, useMLP)
    MultiCross(libc, useMLP)

    # Regression
    LinearSimple2D(libc, useMLP)
    NonLinearSimple2D(libc, useMLP)
    # LinearSimple3D()
    # LinearTricky3D()
    # NonLinearSimple3D()

def Show(title):
    plt.title(title)
    plt.show()
    plt.clf()

