import numpy as np
import matplotlib.pyplot as plt


def euclDistance(vector1, vector2):
    '''
    计算平方欧几里德距离
    :param vector1: 向量1
    :param vector2: 向量2
    :return: 两个向量之间的距离
    '''
    return np.sum(np.power(vector2 - vector1, 2))


def initCentroids(dataSet, k):
    '''
    用随机样本初始化质点

    :param dataSet: 训练数据
    :param k: 需要计算的分类数
    :return: 随机初始化的k个质点
    '''
    numSamples, dim = dataSet.shape  # 将训练数据看作矩阵，求出行数和列数
    centroids = np.zeros((k + 1, dim))  # 初始化一个 (k+1)*dim 的零矩阵
    s = set()  # python中的数据类型：集合
    for i in range(1, k + 1):  # i 从 1 迭代到 k
        while True:
            index = int(np.random.uniform(0, numSamples))  # 将生成的随机数转换成整型
            if index not in s:  # 若集合 s 中不包含 index，则将其加入其中
                s.add(index)
                break
        print("random index: ", index)
        centroids[i, :] = dataSet[index, :]  # 将 dataSet 的第 index 行赋值给 centroids 的第 i 行
    return centroids


def getcost(clusterAssment):
    '''
    获取cost
    :param clusterAssment: 用于存储聚类结果的矩阵
    :return: cost
    '''
    len = clusterAssment.shape[0]  # 获取聚类结果的行数，也就是样本的个数
    Sum = 0.0
    for i in range(len):
        Sum = Sum + clusterAssment[i, 1]  # 叠加样本与质点的距离，得到 cost
    return Sum


def kmeans(dataSet, k):
    '''
    k-means主算法
    :param dataSet: 训练数据
    :param k: 类别个数
    :return: 质点和聚类结果
    '''
    numSamples = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((numSamples, 2)))  # 第一列存这个样本点属于哪个簇，第二列存这个样本点和质点的距离
    for i in range(numSamples):  # 初始化
        clusterAssment[i, 0] = -1
    clusterChanged = True

    # step 1: 初始化 centroids 矩阵，随机获取k个质点
    centroids = initCentroids(dataSet, k)

    while clusterChanged:  # 如果已经收敛，则 clusterChanged 的值为 False
        clusterChanged = False
        for i in range(numSamples):  # 对于每个样本点
            minDist = 100000.0  # 用于记录距离各个质点的最小距离
            minIndex = 0  # 用于记录质点标号

            # step 2: 找到最近的质点
            for j in range(1, k + 1):  # 对于每个质点
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # step 3: 更新样本点与质点的分配关系
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist
            else:
                clusterAssment[i, 1] = minDist

        # step 4: 更新质点
        for j in range(1, k + 1):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    '''
    以2D形式可视化数据
    :param dataSet: 训练数据
    :param k: 类别数
    :param centroids: 样本质点
    :param clusterAssment: 聚类结果
    :return: void
    '''
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

    # 绘制所有非中心样本点
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex - 1])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质点
    for i in range(1, k + 1):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i - 1], markersize=8)

    plt.show()


# step 1: 载入数据
print("step 1: load data...")
dataSet = []
with open('./testSet.txt') as fileIn:
    for line in fileIn.readlines():
        line = line.strip()
        lineArr = line.split(",")
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

# step 2: 开始聚合...
print("step 2: clustering...")
dataSet = np.mat(dataSet)
print("dataSet: ", dataSet)
k = 2
centroids, clusterAssment = kmeans(dataSet, k)

# step 3: 显示结果
print("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment)
