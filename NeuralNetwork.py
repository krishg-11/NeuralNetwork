import re
import sys
import math
import random
import time

r = 1
greater = True

input = sys.argv[1]
regex = re.search(r"(>|<)=?([.0-9]+)", input)

greater, r = regex.groups()
greater = greater==">"
r = float(r)


global transType
transfers = {"T1":lambda x: x, "T2": lambda x: max(0,x), "T3": lambda x: 1/(1+math.exp(-x)), "T4": lambda x: (-1 + 2/(1+math.exp(-x)))}
transDerivs = {"T1": lambda y:1, "T2": lambda y: 1 if y>1 else 0, "T3": lambda y: y*(1-y), "T4": lambda y: (1-y*y)/2}
transType = "T3"

def generateData(r, greater):
    testCases = []
    for i in range(20000):
        x = random.uniform(-1.5,1.5)
        y = random.uniform(-1.5,1.5)
        val = x**2 + y**2
        output = (val>r) == greater
        testCases.append(([x,y,1],[int(output)]))
    return testCases

def dot(a,b):
    return sum([a[i]*b[i] for i in range(len(a))])

def feedforward(inputs, weights):
    global transType
    ffInfo = [inputs]
    currLayer = inputs
    lastLayer = weights[-1]
    for k in range(len(weights)-1):
        layerWeights = weights[k]
        layerOutputs = []
        for j in range(0,len(layerWeights),len(currLayer)):
            layerOutputs.append(transfers[transType](dot(currLayer,layerWeights[j:j+len(currLayer)])))
        ffInfo.append(layerOutputs)
        currLayer = layerOutputs
    finalOutput = [currLayer[i]*lastLayer[i] for i in range(len(currLayer))]
    ffInfo.append(finalOutput)
    return ffInfo

def backProp(weights, ffInfo, output):
    backInfo = [layer.copy() for layer in ffInfo]

    for i in range(len(backInfo[-1])): #last layer *special case*
        backInfo[-1][i] = output[i] - ffInfo[-1][i]
    for i in range(len(backInfo[-2])): #second to last layer *special case*
        backInfo[-2][i] = backInfo[-1][i] * weights[-1][i] * transDerivs[transType](ffInfo[-2][i])
    for k in range(len(backInfo)-3,0,-1):
        for i in range(len(backInfo[k])):
            total = sum([weights[k][i*len(backInfo[k+1])+j] * backInfo[k+1][j] for j in range(len(backInfo[k+1]))])
            backInfo[k][i] = total*transDerivs[transType](ffInfo[k][i])
    return backInfo

def gradientDescent(weights, backInfo, ffInfo, alpha):
    for k in range(len(weights)):
        for i in range(len(weights[k])//len(backInfo[k+1])):
            for j in range(len(backInfo[k+1])):
                weights[k][i+j*len(backInfo[k])] += backInfo[k+1][j] * ffInfo[k][i] * alpha
    return weights

def errorAndAccAll(weights, testCases):
    errorAll = []
    accAll = []
    for test in testCases:
        output = feedforward(test[0],weights)[-1]
        expected = test[1]
        errorAll.append(1/2 * sum([(output[i]-expected[i])**2 for i in range(len(output))]))
        output = [0 if y<0.5 else 1 for y in output]
        accAll.extend([abs(expected[i]-output[i]) for i in range(len(output))])
    return errorAll, accAll


testCases = generateData(r, greater)
layerLens = [3, 5, 3, 1, 1] #HYPERPARAMETER

weights = [[0]*(layerLens[k]*layerLens[k+1]) for k in range(len(layerLens)-1)]

for k in range(len(weights)):
    for i in range(len(weights[k])):
        weights[k][i] = random.uniform(-2,2)
alpha = 0.03   #HYPERPARAMETER

E,acc = errorAndAccAll(weights, testCases)
bestE = sum(E)
bestAcc = sum(acc)/len(acc)
print(bestE)
print(1 - bestAcc)
lastTime = 0

while(True):
    for testCase in range(len(testCases)):
        lastTime += 1
        ffInfo = feedforward(testCases[testCase][0],weights)
        backInfo = backProp(weights, ffInfo, testCases[testCase][1])
        weights = gradientDescent(weights, backInfo, ffInfo, alpha)

        output = ffInfo[-1]
        expected = testCases[testCase][1]
        E[testCase] = 1/2 * sum([(output[i]-expected[i])**2 for i in range(len(output))])
        output = [0 if y<0.5 else 1 for y in output]
        acc[testCase] = abs(expected[0] - output[0])


    newAcc = sum(acc)/len(acc)
    if(newAcc < bestAcc):
        bestAcc = newAcc

    if(sum(E)<bestE):
        bestE = sum(E)
        if(lastTime > 100000):
            print()
            print("error:", sum(E))
            print("Accuracy:", 1-bestAcc)
            print("error", bestE)
            print("Layer cts:", layerLens)
            print("Weights:")
            for x in weights: print(x)

            lastTime = 0
    if(newAcc > 0.95):
        alpha = 0.004
    elif(newAcc > 0.93):
        alpha = 0.006




print(weights)
