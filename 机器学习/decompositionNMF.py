from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition
 
 
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
 
 
###############################################################################
# Load faces data #shuffle洗牌,打乱原始数据的次序
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
print('dataset.target',dataset.target)
faces = dataset.data
 
###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row)) #创建图片,知道图片大小(英寸)
    plt.suptitle(title, size=16)#标题以及字号
 
    for i, comp in enumerate(images):#枚举, 前一个为序号(0~len-1),一个为内容值
        print(i,'~ ',comp)
        plt.subplot(n_row, n_col, i + 1)#选择画制的子图
        vmax = max(comp.max(), -comp.min())
        #对数值归一化,并以灰度图形式显示
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest', vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())#去除子图的坐标轴标签
    #对子图位置及间隔调整
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)
    
 
     
plot_gallery("First centered Olivetti faces", faces[:n_components])
###############################################################################
 
estimators = [
    ('Eigenfaces - PCA using randomized SVD',
         decomposition.PCA(n_components=6,whiten=True)),
 
    ('Non-negative components - NMF',
         decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))
]
 
###############################################################################
 
for name, estimator in estimators:#分别调用PCA和NMF
    print("Extracting the top %d %s..." % (n_components, name))
    print(faces.shape)
    estimator.fit(faces)#调用PCA或NMF提取特征
    components_ = estimator.components_#获取提取特征
    plot_gallery(name, components_[:n_components])
    #按照固定格式进行排序
plt.show()
