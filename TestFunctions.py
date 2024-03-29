import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
nbIteration=250
rbfGamma=0.1
rbfRepresentantProportion=1
rbfMaxIter=1000
class Model(Enum):
    LIN = 1
    MLP = 2
    RBF = 3
multiClassColors = {0: 'lightblue', 1: 'pink', 2: 'lightgreen'}
ax=None
fig=None
def Init(is3D : bool):
    global fig
    if not is3D :
        fig= plt.figure()
    else :
        global ax
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

def Predict(libc, model : Model, isClassification : bool, entries, X, Y, width_size : int, height_size : int, resolution : int, width_offset = 0, height_offset = 0, threeDimensions = False,multiClass = False, depth_size = 0, depth_offset = 0):

    if isClassification:
        if threeDimensions:
            test_X = np.array([[(w / resolution) * width_size + width_offset, (h / resolution) * height_size + height_offset,(z/resolution)*depth_size+depth_offset] for w in range(0, resolution) for h in range(0, resolution) for z in range(0,resolution)], np.float64)
        else:
            test_X = np.array([[(w / resolution) * width_size + width_offset, (h / resolution) * height_size + height_offset] for w in range(0, resolution) for h in range(0, resolution)], np.float64)
    else:
        if threeDimensions:
            #When in regression we sample in one less dimension so in fact it's same x as 2D classification
            test_X = np.array([[(w / resolution) * width_size + width_offset, (h / resolution) * height_size + height_offset] for w in range(0, resolution) for h in range(0, resolution)], np.float64)
        else:
            test_X = np.array([[(w / resolution) * width_size + width_offset] for w in range(0, resolution)], np.float64)
        test_Y = []
    test_colors = []
    shape=np.shape(Y);
    outputSize= 1 if len(shape)==1 else shape[1]
    if model.value == Model.MLP.value:
        idMLP = libc.mlpCreate(entries, entries.size)
        #void Train(const Real* rawAllInputs, Integer inputSize, Integer inputsCount, const Real* rawExcpectedOutputs, Integer outputSize, Integer outpuCount, bool isClassification = true, Real alpha = 0.01f, Integer maxIter = 1000);

        libc.mlpTrain(idMLP, X.ravel(), np.shape(X)[1], np.shape(X)[0], Y.ravel(), outputSize, np.shape(Y)[0], isClassification, 0.1, nbIteration)
        for input_x in test_X:
            raveled = input_x.ravel()
            predictCount = libc.mlpPredict(idMLP, raveled, raveled.size, isClassification)
            f=libc.mlpGetPredictData(idMLP, predictCount)
            if isClassification:
                if multiClass :
                    classes=[libc.mlpGetPredictData(idMLP, n+1) for n in range(predictCount)]
                    colorIndex=max(range(predictCount), key=lambda n:  classes[n])
                    print(colorIndex)
                    print(classes[colorIndex])
                    if colorIndex<0 :
                        if colorIndex>2:
                            print('##Err')
                            print(colorIndex)
                    #for cl in classes:
                        #print(cl)
                    test_colors.append(multiClassColors.get(colorIndex, 'maroon'))
                else:
                    test_colors.append('lightblue' if f >= 0 else 'pink')
            else:
                test_Y.append(f)
                test_colors.append('lightblue')
        libc.mlpDelete(idMLP)
    elif model.value == Model.LIN.value:
        idLinear = libc.linearCreate(isClassification,0.01, np.shape(X)[1])
        libc.linearTrain(idLinear, nbIteration,X.ravel(), Y.ravel(), np.shape(X)[0])
        test_colors = ['lightblue' if libc.linearEvaluate(idLinear, input_x) >= 0 else 'pink' for input_x in test_X]
        libc.linearDelete(idLinear)
    elif model.value == Model.RBF.value:
        idRbf = libc.rbfCreate(rbfGamma)
        #rbfTrain(TypeId id, Integer sizeOfModel,const Real* rawAllInputs, Integer numberOfInputSubArray, Integer sizeOfInputSubArray, const Real* matrixOutputRowAligned, Integer sizeOfRow, Integer numberOfRow);
        libc.rbfTrain(idRbf, int(np.floor(rbfRepresentantProportion*np.shape(X)[0])), X.ravel(), np.shape(X)[0], np.shape(X)[1], Y.ravel(), outputSize, np.shape(Y)[0], rbfMaxIter)
        #Real rbfPredict(TypeId id,bool isClassification, const Real* rawInputs, Integer rawInputsCount);
        arr=[libc.rbfPredict(idRbf, isClassification, input_x.ravel(), input_x.ravel().size) for input_x in test_X]
        if isClassification :
            test_colors = ['lightblue' if f>= 0 else 'pink' for f in arr]
        else :
            test_Y = arr
            test_colors = ['lightblue' for n in range(len(arr))]
# Show prediction
    if isClassification:
        if threeDimensions :
            ax.scatter(test_X[:, 0], test_X[:, 1], test_X[:, 2] , c=test_colors)
        else :
            plt.scatter(test_X[:, 0], test_X[:, 1], c=test_colors)
    else:
        if threeDimensions:
            ax.scatter(test_X[:, 0], test_X[:, 1], test_Y, c=test_colors)
        else:
            plt.scatter(test_X, test_Y, c=test_colors)

def LinearSimple(libc, model, width_size=4, height_size=4, resolution=100):

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
    Predict(libc, model, True, entries, X, Y, width_size, height_size, resolution)

    # Show dataset
    plt.scatter(X[0, 0], X[0, 1], color='blue')
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red')
    Show('Linear simple', model)

def LinearMultiple(libc, model, width_size=2, height_size=2, resolution=100, width_offset = 1, height_offset = 1):
    X = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    X.astype(np.float64)
    Y.astype(np.float64)

    entries = np.array([2, 1], np.int32)
    Predict(libc, model, True, entries, X, Y, width_size, height_size, resolution, width_offset, height_offset)

    plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='red')
    Show('Linear multiple', model)


def XOR(libc, model, width_size=1, height_size=1, width_points=100, resolution=100):
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], np.float64)
    Y = np.array([1, 1, -1, -1], np.float64)

    entries = np.array([2, 2, 1], np.int32)
    Predict(libc, model, True, entries, X, Y, width_size, height_size,  resolution)

    plt.scatter(X[0:2, 0], X[0:2, 1], color='blue')
    plt.scatter(X[2:4, 0], X[2:4, 1], color='red')
    Show('XOR', model)


def Cross(libc, model, width_size=2, height_size=2, resolution=100, width_offset = -1, height_offset = -1):
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X], np.float64)

    X.astype(np.float64)

    entries = np.array([2, 4, 1], np.int32)
    Predict(libc, model, True, entries, X, Y, width_size, height_size, resolution, width_offset, height_offset)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 1],
                color='red')
    Show('Cross', model)


def MultiLinear3Classes(libc, model, width_size=2, height_size=2, resolution=100, width_offset = -1, height_offset = -1):
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
    Predict(libc, model, True, entries, X, Y, width_size, height_size, resolution, width_offset, height_offset,False,True)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    Show('Multi linear 3 classes', model)


def MultiCross(libc, model, width_size=2, height_size=2, resolution=100, width_offset = -1, height_offset = -1):
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 < abs(p[1] % 0.5) else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 >= abs(p[1] % 0.5) else [0, 0, 1] for p in X], np.float64)

    X.astype(np.float64)

    # Note: L'exemple stipule MLP (2, ?, ?, 3)... good luck
    entries = np.array([2, 100, 100, 3], np.int32)
    Predict(libc, model, True, entries, X, Y, width_size, height_size, resolution, width_offset, height_offset,False,True)

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    Show('Multicross', model)


# Regression

def LinearSimple2D(libc, model, width_size=2, height_size=3, resolution=100):
    X = np.array([
        [1],
        [2]
    ], np.float64)
    Y = np.array([
        2,
        3
    ], np.float64)

    entries = np.array([1, 1], np.int32)
    Predict(libc, model, False, entries, X, Y, width_size, height_size, resolution)

    plt.scatter(X, Y)
    Show('Linear Simple 2D', model)


def NonLinearSimple2D(libc, model, width_size=3, height_size=3, resolution=100):
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
    Predict(libc, model, False, entries, X, Y, width_size, height_size, resolution)

    plt.scatter(X, Y)
    Show('Non Linear Simple 2D', model)


def LinearSimple3D(libc, model, width_size=3, height_size=3, depth_size=3, resolution=100):
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

    # Should be [2 ,1], but this leads to a crash...
    entries = np.array([2, 1], np.int32)
    Predict(libc, model, False, entries, X, Y, width_size, height_size, resolution, 0, 0, True, depth_size)

    ax.scatter(X[:, 0], X[:, 1], Y)
    Show('Linear Simple 3D', model)


def LinearTricky3D(libc, model, width_size=3, height_size=3, depth_size=3, resolution=100):
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

    # Should be [2 ,1], but this leads to a crash...
    entries = np.array([2, 1], np.int32)
    Predict(libc, model, False, entries, X, Y, width_size, height_size, resolution, 0, 0, True, depth_size)

    ax.scatter(X[:, 0], X[:, 1], Y)
    Show('Linear tricky 3D', model)


def NonLinearSimple3D(libc, model, width_size=1, height_size=1, depth_size=1, resolution=100):
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

    # Should be [2, 2, 1] but this leads to a crash
    entries = np.array([2, 2, 1], np.int32)
    Predict(libc, model, False, entries, X, Y, width_size, height_size, resolution, 0, 0, True, depth_size)

    ax.scatter(X[:, 0], X[:, 1], Y)
    Show('Non Linear Simple 3D', model)

def Show(title, model):
    plt.title(title + ' ' + model.name)
    plt.show()
    plt.clf()