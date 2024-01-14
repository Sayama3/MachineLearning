import matplotlib.pyplot as plt
import numpy as np

def Show(title):
    plt.title(title)
    plt.show()
    plt.clf()


def LinearSimple(libc, useMLP = False, width_size=4, height_size=4, width_points=100, height_points=100):

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

    test_X = np.array([[(w / width_points) * width_size, (h / height_points) * height_size] for w in range(0, width_points) for h in range(0, height_points)], np.float64)
    test_colors = []

    if useMLP:
        idMLP = libc.mlpCreate(entries, entries.size)
        libc.mlpTrain(idMLP, X.ravel(), np.shape(X)[1], np.shape(X)[0], Y.ravel(), 1, np.shape(Y)[0], True, 1, 1000)
        for input_x in test_X:
            predictCount = libc.mlpPredict(idMLP, input_x.ravel(), input_x.ravel().size, False)
            test_colors.append('pink' if libc.mlpGetPredictData(idMLP, 0) >= 0 else 'lightblue')
        libc.mlpDelete(idMLP)

    else:
        idLinear = libc.linearCreate(1, X.ravel(), Y.ravel(), np.shape(X)[1], np.shape(X)[0])
        libc.linearTrain(idLinear, 5000, 0)
        test_colors = ['pink' if libc.linearEvaluate(idLinear, input_x) >= 0 else 'lightblue' for input_x in test_X]
        libc.linearDelete(idLinear)

    # Show prediction
    plt.scatter(test_X[:, 0], test_X[:, 1], c=test_colors)

    # Show dataset
    plt.scatter(X[0, 0], X[0, 1], color='blue')
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red')
    Show('Linear simple')


def LinearMultiple():
    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])

    plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='red')
    Show('Linear multiple')


def XOR():
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])

    plt.scatter(X[0:2, 0], X[0:2, 1], color='blue')
    plt.scatter(X[2:4, 0], X[2:4, 1], color='red')
    Show('XOR')


def Cross():
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 1],
                color='red')
    Show('Cross')


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
    Show('Multi linear 3 classes')


def MultiCross():
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 < abs(p[1] % 0.5) else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 >= abs(p[1] % 0.5) else [0, 0, 1] for p in X])

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
    Show('Linear Simple 2D')


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
    Show('Non Linear Simple 2D')


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
    Show('Linear Simple 3D')


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
    Show('Linear tricky 3D')


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
    Show('Non Linear Simple 3D')


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
