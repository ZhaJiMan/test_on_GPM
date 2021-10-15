#------------------------------------------------------------------------------
# 2021/07/26
# 修改后的连通域标记算法,支持搜索半径以内的像素作为邻居.
# 有Seed-Filling和Two-Pass两种实现.
#------------------------------------------------------------------------------
import numpy as np
from scipy.stats import rankdata

def get_neighbor_indices(labelled, row, col, r):
    '''找出一点在半径r范围内labelled值等于-1的点的下标.'''
    nrow, ncol = labelled.shape
    for i in range(-r, r + 1):
        k = r - abs(i)
        for j in range(-k, k + 1):
            # 跳过这个点本身.
            if i == 0 and j == 0:
                continue
            # 将下标偏移值加给row和col.
            x = row + i
            y = col + j
            # 避免下标出界.
            if x >= 0 and x < nrow and y >= 0 and y < ncol:
                if labelled[x, y] == -1:
                    yield x, y

def seed_filling(image, radius=1):
    '''
    用Seed-Filling算法寻找二值图像里的连通域.

    Parameters
    ----------
    image : ndarray, shape (nrow, ncol)
        二维整型数组,0代表图像的背景,非零值代表前景.

    radius : int
        以radius为半径搜索当前像素周围的邻居像素.

    Returns
    -------
    labelled : ndarray, shape (nrow, ncol)
        二维整型数组,元素的数值表示所属连通域的标号.
        0表示背景,从1开始表示不同的连通域.

    nlabel : int
        图像中连通域的个数.
    '''
    # 检测radius是否为大于0的整数.
    if not (isinstance(radius, (int, np.integer)) and radius > 0):
        raise ValueError('radius should be an integer greater than 0')

    # 用-1表示图像上还未被标记的特征.
    labelled = np.where(image != 0, -1, 0).astype(int)
    nrow, ncol = labelled.shape
    label = 1

    for row in range(nrow):
        for col in range(ncol):
            # 跳过背景和已经被标记过的特征像素.
            if labelled[row, col] != -1:
                continue
            labelled[row, col] = label
            neighbor_indices = list(
                get_neighbor_indices(labelled, row, col, radius)
            )
            # 不断寻找邻居像素的邻居并标记之,直至再找不到未被标记的邻居.
            while neighbor_indices:
                neighbor_index = neighbor_indices.pop()
                labelled[neighbor_index] = label
                for new_index in get_neighbor_indices(
                    labelled, *neighbor_index, radius
                ):
                    neighbor_indices.append(new_index)
            label += 1

    return labelled, label - 1

class UnionFind:
    '''用列表实现简单的并查集.'''
    def __init__(self, n):
        '''创建含有n个节点的并查集,每个元素指向自己.'''
        self.parents = list(range(n))

    def find(self, i):
        '''递归查找第i个节点的根节点,同时压缩路径.'''
        parent = self.parents[i]
        if parent == i:
            return i
        else:
            root = self.find(parent)
            self.parents[i] = root
            return root

    def union(self, i, j):
        '''合并节点i和j所属的两个集合.保证大的根节点被合并到小的根节点上.'''
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i < root_j:
            self.parents[root_j] = root_i
        elif root_i > root_j:
            self.parents[root_i] = root_j
        else:
            return None

def get_neighbor_labels(labelled, row, col, r):
    '''找出一点在半径r范围内值大于0的label.'''
    nrow, ncol = labelled.shape
    # 只选取上边和左边的下标.
    for i in range(-r, 1):
        k = r + i
        if i == 0:
            j0, j1 = -k, -1
        else:
            j0, j1 = -k, k
        for j in range(j0, j1 + 1):
            # 将下标偏移值加给row和col.
            x = row + i
            y = col + j
            # 避免下标出界.
            if x >= 0 and x < nrow and y >= 0 and y < ncol:
                neighbor_label = labelled[x, y]
                if neighbor_label > 0:
                    yield neighbor_label

def two_pass(image, radius=1):
    '''
    用Two-Pass算法寻找图片里的连通域.

    Parameters
    ----------
    image : ndarray, shape (nrow, ncol)
        二维整型数组,零值代表背景,非零值代表特征.

    radius : int
        以radius为半径搜索当前像素周围的邻居像素.

    Returns
    -------
    labelled : ndarray, shape (nrow, ncol)
        二维整型数组,元素的数值表示所属连通域的标号.
        0表示背景,从1开始表示不同的连通域.

    nlabel : int
        图像中连通域的个数.
    '''
    nrow, ncol = image.shape
    labelled = np.zeros_like(image, dtype=int)
    uf = UnionFind(image.size // 2)
    label = 1

    # 第一遍循环,用label标记出连通的区域.
    for row in range(nrow):
        for col in range(ncol):
            # 跳过背景.
            if image[row, col] == 0:
                continue
            # 若当前像素周围没有label大于零的像素,则该像素获得新label.
            # 否则用并查集记录相邻像素的label间的关系.
            neighbor_labels = list(
                get_neighbor_labels(labelled, row, col, radius)
            )
            if len(neighbor_labels) == 0:
                labelled[row, col] = label
                label += 1
            else:
                first_label = neighbor_labels[0]
                labelled[row, col] = first_label
                for neighbor_label in neighbor_labels[1:]:
                    uf.union(first_label, neighbor_label)

    # 获取代表每个集合的label,并利用大小排名重新赋值.
    roots = [uf.find(i) for i in range(label)]
    labels = rankdata(roots, method='dense') - 1
    # 第二遍循环赋值利用ndarray的advanced indexing实现.
    labelled = labels[labelled]

    return labelled, labelled.max()

# 测试.
if __name__ == '__main__':
    # image = np.zeros((7000, 50), dtype=int)
    # image[5000:6000, :] = np.random.randint(0, 2, (1000, 50))
    image = np.random.randint(0, 2, (1000, 50))
    # labelled1 = seed_filling(image, radius=4)
    # labelled2 = two_pass(image, radius=4)
