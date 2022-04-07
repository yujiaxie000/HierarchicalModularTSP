import numpy as np
from hmNetwork import *
from plotter import *
import csv
import json
from convertor import *
SCALE = 1.0

def getStructure(layerNum, nodePerModule):
    hierarchy = []
    currLayer = [nodePerModule]
    total = 0
    for i in range(layerNum):
        curr = []
        temp = []
        for j in range(len(currLayer)):
            layer = generateLayers(currLayer[j], nodePerModule)
            curr.extend(layer)
            temp.append(layer)
        hierarchy.append(temp)
        currLayer = curr
    return sum(currLayer), hierarchy

def generateLayers(moduleNum, nodePerModule):
    sizes = np.random.normal(loc=nodePerModule, scale=SCALE, size=moduleNum).astype(int)
    sizes = np.where(sizes<=0, nodePerModule, sizes)
    return sizes
    


def generate(layerNum, nodePerModule, figoutFile, nodeoutFile, logbase=3):
    nodeNum, hierarchy = getStructure(layerNum, nodePerModule)
    # print(nodeNum)

    hierarchy2D, trace = assignDistance(hierarchy, logbase)
    x_minmax, y_minmax = get_minmax(hierarchy2D)
    h2D = []
    for h in hierarchy2D:
        temp = np.array(h).astype(int)
        if x_minmax[0] < 0:
            temp[:,1] -= int(x_minmax[0])
        if y_minmax[0] < 0:
            temp[:,2] -= int(y_minmax[0])
        h2D.append(temp)

    # print(h2D[-1])

    # np.save(nodeoutFile + ".npz", *h2D)
    np.save(nodeoutFile + ".npy", np.vstack(h2D))
    traceJson = json.dumps(trace)
    with open(nodeoutFile + ".json", "w", newline="\n", encoding="utf-8") as fout:
        fout.write(traceJson)

    
    # lastLayer = h2D[-1]
    # idx = np.arange(nodeNum) + 1
    # lastLayer = np.concatenate((idx.reshape(-1,1), lastLayer), 1)
    # instance = convert(h2D[1].astype(int), nodeoutFile + ".vrp.txt")

    # print(lastLayer)
        
    plot_hierarchy(hierarchy2D, figoutFile)
    # plot_hierarchy_split(hierarchy2D, figoutFile)

    # HM = HierarchicalModularNetwork(nodeNum, hierarchy)
    # HM.structureAnalysis_nx(HM.root, len(hierarchy) - 2)
    # HM.structuralAnalysis_ig(HM.root, len(hierarchy) - 1)
    return nodeNum

def assignDistance(hierarchy, logbase):
    dist = [logbase ** (i+3) for i in range(len(hierarchy) + 1)]
    newHierarchy = []
    currCenters = [np.array([0,0,0])]
    hierarchyTrace = {0: (0, 0)}

    idx = 0
    traceIdx = 1
    for i in range(len(dist)-1, 0, -1):
        newHierarchy.append(currCenters)
        currCenters, traceIdx = processLayer(currCenters, hierarchy[idx], (dist[i], dist[i]/(logbase ** 2)), False, traceIdx, hierarchyTrace)
        idx += 1
    newHierarchy.append(currCenters)
    currCenters, traceIdx = processLayer(currCenters, hierarchy[-1], (dist[0], dist[0]/logbase), True, traceIdx, hierarchyTrace)
    newHierarchy.append(currCenters)

    return newHierarchy, hierarchyTrace

def processLayer(centers, layer, radius, isLast, traceIdx, hierarchyTrace):
    nextLayer = []
    currTraceID = traceIdx
    if isLast:
        idx = 0
        for arr in layer:
            for item in arr:
                locations = assignLocation(centers[idx], item, radius, isLast)
                locationID = np.arange(currTraceID, currTraceID + len(locations)).reshape(-1,1)
                hierarchyTrace[int(centers[idx][0])] = locationID.flatten().tolist()
                nextLayer.extend(np.hstack((locationID, locations)))
                idx += 1
                currTraceID += len(locations)
    else:
        for i in range(len(centers)):
            locations = assignLocation(centers[i], layer[i], radius, isLast)
            locationID = np.arange(currTraceID, currTraceID + len(locations)).reshape(-1,1)
            hierarchyTrace[int(centers[i][0])] = locationID.flatten().tolist()
            nextLayer.extend(np.hstack((locationID, locations)))
            currTraceID += len(locations)
    return nextLayer, currTraceID

def assignLocation(center, extensions, radius, isLast):
    if isLast:
        radius = np.random.normal(loc=radius[0], scale=radius[1], size=extensions)
        pi = [i/extensions * 2 * np.pi for i in range(extensions)]
    else:
        radius = np.random.normal(loc=radius[0], scale=radius[1], size=len(extensions))
        pi = [i/len(extensions) * 2 * np.pi for i in range(len(extensions))]

    x = np.multiply(radius, np.cos(pi)) + center[1]
    y = np.multiply(radius, np.sin(pi)) + center[2]
    return [(int(x[i]), int(y[i])) for i in range(len(x))]

# def getModules(nodeList, nodePerModule):
#     total = len(nodeList)
#     sizes = np.random.normal(loc=nodePerModule, scale=SCALE, size=int(total//nodePerModule + 1)).astype(int)
#     sizes = np.where(sizes<=0, nodePerModule, sizes)
#     splitIdx = getSplitIdx(sizes, total)
#     modules = np.split(nodeList, splitIdx)
#     print(len(modules))
    
# def getSplitIdx(sizeList, total):
#     splitIdx = np.zeros(len(sizeList))
#     for i in range(len(splitIdx) - 1):
#         splitIdx[i+1] = splitIdx[i] + sizeList[i]
#     splitIdx = np.array(splitIdx)
#     return splitIdx[splitIdx < total][1:].astype(int)


if __name__ == "__main__":
    nodeNum = generate(2, 3, "Figures/TSP2D", "Instances/nodes1", logbase=4)

    # generate(nodeNum, 3, 40)