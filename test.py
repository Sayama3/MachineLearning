import ctypes.wintypes
import os

import TestFunctions

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


# Main

dlls = find_dll()
if len(dlls) == 0:
    print("Couldn't find dll named 'MLCore.dll' nor 'libMLCore.dll'.")
    exit()

dllPath = dlls[0]
print("Using '" + dllPath + "'.")
libc = cdll.LoadLibrary(dllPath)

# Function prototypes
INT = ctypes.wintypes.INT
REAL = ctypes.c_double
BOOL = ctypes.c_bool

ND_POINTER_INT = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C")
ND_POINTER_FLOAT = np.ctypeslib.ndpointer(REAL, ndim=1, flags="C")

libc.mlpCreate.argtypes = [ND_POINTER_INT, INT]
libc.mlpCreate.restype = INT

#	ML_API void mlpPropagate(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification);
# libc.mlpPropagate.argtypes = [INT,ND_POINTER_FLOAT,INT,ctypes.c_bool]

#void mlpTrain(TypeId id, const Real* rawAllInputs, Integer rawAllInputsWidth, Integer rawAllInputsHeight, const Real* rawExcpectedOutputs, Integer rawExcpectedOutputsWidth, Integer rawExcpectedOutputsHeight, bool isClassification = true, float alpha = 0.01f, Integer maxIter = 1000);

libc.mlpTrain.argtypes = [INT, ND_POINTER_FLOAT, INT, INT, ND_POINTER_FLOAT, INT, INT, ctypes.c_bool, REAL, INT]

# 	ML_API Real mlpPredict(TypeId id, const Real* rawInputs, Integer rawInputsCount, bool isClassification = true);
libc.mlpPredict.argtypes = [ctypes.wintypes.INT, ND_POINTER_FLOAT, ctypes.wintypes.INT, BOOL]
libc.mlpPredict.restype = INT

# 	ML_API Real mlpGetPredictData(TypeId id, Integer index);
libc.mlpGetPredictData.argtypes = [ctypes.wintypes.INT, ctypes.wintypes.INT]
libc.mlpGetPredictData.restype = ctypes.wintypes.DOUBLE
#ML_API TypeId linearCreate(bool isClassification,Real step,Integer entrySize);
libc.linearCreate.argtypes = [BOOL, REAL, INT]
libc.linearCreate.restype = INT

#ML_API void linearTrain(TypeId id,Integer count,const Real* entries, const Real* output, Integer entryCount);
libc.linearTrain.argtypes = [INT, INT,ND_POINTER_FLOAT, ND_POINTER_FLOAT, INT ]

#     ML_API Real linearEvaluate(TypeId id,const Real* entries);
libc.linearEvaluate.argtypes = [INT, ND_POINTER_FLOAT]
libc.linearEvaluate.restype = REAL

libc.linearDelete.argtypes = [INT]

libc.initialize()

# Execute test functions
#Classification
#TestFunctions.nbIteration=500
#TestFunctions.XOR(libc, True)
#TestFunctions.LinearSimple(libc, True)
#TestFunctions.nbIteration=1000
#TestFunctions.LinearMultiple(libc, True)
#TestFunctions.nbIteration=25000
#TestFunctions.Cross(libc,True)
#TestFunctions.MultiCross(libc,False)
#TestFunctions.nbIteration=1000
#TestFunctions.LinearSimple2D(libc, True)
TestFunctions.nbIteration=5000
TestFunctions.LinearSimple3D(libc, True)
#TestFunctions.LinearSimple3D(libc, False)
#TestFunctions.MultiLinear3Classes(libc,False)

libc.shutdown()

# Plotting functions
# Classification
