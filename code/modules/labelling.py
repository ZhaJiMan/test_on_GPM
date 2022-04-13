import numpy as np
from scipy.stats import rankdata

def seed_filling(image, radius=1):
    '''
    用Seed-Filling算法标记图片中的连通域.

    Parameters
    ----------
    image : ndarray, shape (nrow, ncol)
        图片数组, 零值表示背景, 非零值表示特征.

    radius : int
        以当前像素为原点, radius为半径的圆形区域为邻域.

    Returns
    -------
    labelled : ndarray, shape (nrow, ncol), dtype int
        表示连通域标签的数组, 0表示背景, 从1开始表示标签.

    nlabel : int
        连通域的个数.
    '''
    # 检测radius是否为大于0的整数.
    if not (isinstance(radius, (int, np.integer)) and radius > 0):
        raise ValueError('radius should be an integer greater than 0')

    # 用-1表示未被标记过的特征像素.
    image = np.asarray(image, dtype=bool)
    nrow, ncol = image.shape
    labelled = np.where(image, -1, 0)

    # 指定邻域的范围.
    offsets = []
    for i in range(-radius, radius + 1):
        k = radius - abs(i)
        for j in range(-k, k + 1):
            offsets.append((i, j))
    offsets.remove((0, 0))  # 去掉原点.

    def get_neighbor_indices(row, col):
        '''获取(row, col)位置邻域的下标.'''
        for (dx, dy) in offsets:
            x = row + dx
            y = col + dy
            if x >= 0 and x < nrow and y >= 0 and y < ncol:
                yield x, y

    label = 1
    for row in range(nrow):
        for col in range(ncol):
            # 跳过背景像素和已经标记过的特征像素.
            if labelled[row, col] != -1:
                continue
            # 标记当前位置和邻域内的特征像素.
            current_indices = []
            labelled[row, col] = label
            for neighbor_index in get_neighbor_indices(row, col):
                if labelled[neighbor_index] == -1:
                    labelled[neighbor_index] = label
                    current_indices.append(neighbor_index)
            # 不断寻找与特征像素相邻的特征像素并标记, 直至再找不到特征像素.
            while current_indices:
                current_index = current_indices.pop()
                labelled[current_index] = label
                for neighbor_index in get_neighbor_indices(*current_index):
                    if labelled[neighbor_index] == -1:
                        labelled[neighbor_index] = label
                        current_indices.append(neighbor_index)
            label += 1

    return labelled, label - 1

class UnionFind:
    '''用列表实现简单的并查集.'''
    def __init__(self, n):
        '''创建含有n个节点的并查集, 每个元素指向自己.'''
        self.parents = list(range(n))

    def find(self, i):
        '''递归查找第i个节点的根节点, 同时压缩路径.'''
        parent = self.parents[i]
        if parent == i:
            return i
        else:
            root = self.find(parent)
            self.parents[i] = root
            return root

    def union(self, i, j):
        '''
        合并节点i和j所属的两个集合.
        保证大的根节点被合并到小的根节点上, 以配合连通域算法的循环顺序.
        '''
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i < root_j:
            self.parents[root_j] = root_i
        elif root_i > root_j:
            self.parents[root_i] = root_j
        else:
            return None

def two_pass(image, radius=1):
    '''
    用Two-Pass算法标记图片中的连通域.

    Parameters
    ----------
    image : ndarray, shape (nrow, ncol)
        图片数组, 零值表示背景, 非零值表示特征.

    radius : int
        以当前像素为原点, radius为半径的圆形区域为邻域.

    Returns
    -------
    labelled : ndarray, shape (nrow, ncol), dtype int
        表示连通域标签的数组, 0表示背景, 从1开始表示标签.

    nlabel : int
        连通域的个数.
    '''
    image = np.asarray(image, dtype=bool)
    nrow, ncol = image.shape
    labelled = np.zeros_like(image, dtype=int)
    uf = UnionFind(image.size // 2)

    # 指定邻域的范围, 相比seed-filling只有半边.
    offsets = []
    for i in range(-radius, 1):
        k = radius + i
        j0 = -k
        j1 = -1 if i == 0 else k
        for j in range(j0, j1 + 1):
            offsets.append((i, j))

    def get_neighbor_indices(row, col):
        '''获取(row, col)位置邻域的下标.'''
        for (dx, dy) in offsets:
            x = row + dx
            y = col + dy
            if x >= 0 and x < nrow and y >= 0 and y < ncol:
                yield x, y

    label = 1
    for row in range(nrow):
        for col in range(ncol):
            # 跳过背景像素.
            if not image[row, col]:
                continue
            # 寻找邻域内特征像素的标签.
            feature_labels = []
            for neighbor_index in get_neighbor_indices(row, col):
                neighbor_label = labelled[neighbor_index]
                if neighbor_label > 0:
                    feature_labels.append(neighbor_label)
            # 当前位置取邻域内的标签, 同时记录邻域内标签间的关系.
            if feature_labels:
                first_label = feature_labels[0]
                labelled[row, col] = first_label
                for feature_label in feature_labels[1:]:
                    uf.union(first_label, feature_label)
            # 若邻域内没有特征像素, 当前位置获得新标签.
            else:
                labelled[row, col] = label
                label += 1

    # 获取所有集合的根节点, 由大小排名得到标签值.
    roots = [uf.find(i) for i in range(label)]
    labels = rankdata(roots, method='dense') - 1
    # 利用advanced indexing替代循环修正标签数组.
    labelled = labels[labelled]

    return labelled, labelled.max()
