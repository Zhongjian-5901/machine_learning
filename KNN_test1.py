import numpy as np
import operator

def createDataSet():
    # 四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #四组特征的标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels

if __name__ == '__main__':
    # 创建数据集
    group,labels = createDataSet()
    # 打印数据集
    print(group)  # 数据集
    print(labels)  # 标签分类
'''
Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labels - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
'''

def classify0(inX,dataSet,labLels,k):
    # inX:测试集  dataSet: 训练集
    dataSetSize = dataSet.shape[0]  # shape[0]表示行，[1]表示列
    # 行列复制
    # diffmat = np.tile(inX,(dataSetSize,1))-dataSet
    diffmat = inX - dataSet   #广播机制
    # np.tile函数：按照某个方向复制元素，np.tile([101,20],(4,1))的意思是将[101,20]在行方向上复制4行，列方相上复制1行
    # 二维特征相减后的平方
    sqDiffMat = diffmat**2
    # sum(0)按列相加，sum(1)按行相加
    Distances = sqDiffMat.sum(axis=1)**0.5
    # argsort():返回distance按从小到大排序后的索引值  argsort()[::-1]从大到小
    sortedDistanceIndices = Distances.argsort()
    # 记录前k个中每个类别出现的次数
    # 定义一个字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        votelabels = labels[sortedDistanceIndices[i]]
        # 统计每个类别出现的次数
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        if votelabels not in classCount.keys():
            classCount[votelabels] = 0
        classCount[votelabels] += 1
        classCount[votelabels] = classCount.get(votelabels,0) + 1
    # dict.items() 以列表返回可遍历的(键, 值)元组数组
    # sort() 对列表排序， sorted()对所有可迭代的对象  reverse 降序排序
    # nameRank = sorted(rankList.items(), key = operator.itemgetter(1))
    # rankList 是一个字典，rankList.items() ，Python 字典 items() 方法以列表返回可遍历的(键, 值) 元组数组，就是这样 [(),()]
    # sorted（iterable,function，reverse）,上面的operator.itemgetter(1)表示一个函数，1表示取iterable里的第二个数，用这个数来进行排序。
    sortedClasscount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别=预测类别
    return sortedClasscount[0][0]

if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [101,20]
    test_class = classify0(test,group,labels,3)
    print('测试集分类为',test_class)


''' 广播机制
inx1 = np.array([[1,101],[5,89],[108,5],[115,8]])
data1 = np.array([0,10])
diff1 = inx1 - data1
sq1 = diff1**2
dis1 = sq1.sum(1)
dis = dis1**0.5
print(dis)
'''


