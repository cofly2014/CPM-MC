import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd


S_X1=np.random.random((200,200))#原始特征

class_num=5  

###################

maker=['o','v','^','s','p','*','<','>','D','d','h','H']#设置散点形状
colors = ['black','tomato','yellow','cyan','blue', 'lime', 'r', 'violet','m','peru','olivedrab','hotpink']#设置散点颜色

Label_Com = ['S-1', 'T-1', 'S-2', 'T-2', 'S-3',
             'T-3', 'S-4', 'T-4','S-5','T-5', 'S-6', 'T-6', 'S-7','T-7','S-8', 'T-8','S-9','T-9',
             'S-10','T-10','S-11', 'T-11', 'S-12','T-12'] ##图例名称






def visual(X):
    tsne = manifold.TSNE(n_components=2,init='random', n_iter=9000)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    #'''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return  X_norm


def plot_with_labels(S_lowDWeights,Trure_labels,name):
    plt.cla()#清除当前图形中的当前活动轴,所以可以重复利用

    # 降到二维了，分别给x和y
    True_labels=Trure_labels.reshape((-1,1))
    
    S_data=np.hstack((S_lowDWeights,True_labels))
    S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})
    # ax=plt.axes(projection='3d')


    for index in range(class_num):
        X= S_data.loc[S_data['label'] == index]['x']
        Y=S_data.loc[S_data['label'] == index]['y']
        # Z=S_data.loc[S_data['label'] == index]['z']
        # ax.scatter3D(X, Y, Z,marker=maker[0], c=colors[index])  
        plt.scatter(X,Y, marker=maker[0], c=colors[index],alpha=0.65)

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    #
    plt.title(name,fontsize=32,fontweight='normal',pad=20)



def draw(x,name,num,labels=[]):
    plt.clf()
    #####生成标签
    y_s=np.zeros((num))
    if len(labels)==0:
        for i in range(num):
            y_s[i] =int(i// (num//5))
    else:
        for i in range(num):
            y_s[i] =int(labels[int(i//1)])
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    plot_with_labels(visual(x),y_s,name)


    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None,
                    wspace=0.1, hspace=0.15)
    # plt.legend(scatterpoints=1,labels = Label_Com, loc='best',labelspacing=0.4,columnspacing=0.4,markerscale=2,bbox_to_anchor=(0.9, 0),ncol=12,prop=font1,handletextpad=0.1)

    plt.savefig('./'+name+'.png', format='png',dpi=300, bbox_inches='tight')
    plt.clf()
    # plt.show(fig)

# draw(S_X1,"A",200)

def drawmat(x,name):
    plt.clf()
    plt.imshow(x)  
    plt.colorbar()
    plt.savefig('./'+name+'.png', format='png',dpi=300, bbox_inches='tight')
    plt.clf()
# drawmat(S_X1)

