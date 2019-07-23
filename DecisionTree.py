from math import log
import operator

#计算香浓熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for fectVec in dataSet:
        currentLabel = fectVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataset = [[1,1,'maybe'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataset, labels

#myDat, Labels=createDataSet()

#划分数据集,axis代表第几列,value代表条件，把符合条件的数据挑选出来,本质就是从list中剔除每个数据第axis列值为value的元素,剩下的数据组成新的数据集
def splitDataSet(dataset,axis,value):
    retDataSet = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedFeatVec = featvec[:axis]
            reducedFeatVec.extend(featvec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#print(splitDataSet(myDat,1,1))
#最佳划分方式

def chooseBestFetureToSplit(dataset):
    numFeatures = len(dataset[0])-1
    #print(numFeatures)
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featlist = [example[i] for example in dataset]
        uniqueVals = set(featlist)
        newEntrop = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset,i,value)#划分之后的数据
            prob = len(subDataSet)/float(len(dataset))#选择该分类的概率
            newEntrop += prob * calcShannonEnt(subDataSet)#新的信息熵
        infoGain = baseEntropy - newEntrop#信息增益
        if(infoGain > bestInfoGain):#选择信息增益大的特征作为最好的划分选择
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#myDat, labels = createDataSet()
#print(chooseBestFetureToSplit(myDat))
#投票表决
def majorityCnt(classList):
    classCount={}#创建键值为classList中唯一值的数据字典
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1 #统计每个类标签出现的频率
    sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #利用键值对字典排序，返回出现次数最多的那个
    return sortedclassCount[0][0]

#创建树
def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):#递归结束条件1：所有的类标签完全相同，则直接返回类标签
        return classList[0]
    if len(dataset[0]) == 1:#递归结束条件2：用完了所有的特征，仍然不能把数据集划分成仅包含唯一类别的分组
        return majorityCnt(classList)
    bestFeature = chooseBestFetureToSplit(dataset)#最佳划分特征
    bestFeatureLabels = labels[bestFeature]#
    myTree = {bestFeatureLabels:{}}
    del(labels[bestFeature])
    featureValues = [example[bestFeature] for example in dataset]#获得列表包含所有属性
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabels][value] = createTree(splitDataSet(dataset,bestFeature,value),subLabels)
    return myTree


#分类器
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0] #获取属性
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
                classLabel = classify(secondDict[key],featLabels,testVec)
            else : classlabel = secondDict[key]
    return classLabel

#保存决策树模型
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()
#打开决策树模型
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)


if __name__ == '__main__':
  fr = open('./lenses.txt')
  lenses = [inst.strip().split('\t') for inst in fr.readlines()]
  lensesLabels = ['age','prescript','astigmatic','tearRate']
  #lensesTree = createTree(lenses, lensesLabels)
  print(createTree(lenses,lensesLabels))