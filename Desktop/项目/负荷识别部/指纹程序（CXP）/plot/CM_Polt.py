def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib


    cm = confusion_matrix(y, yp)  # 混淆矩阵
    norm = matplotlib.colors.Normalize(vmin=np.min(cm), vmax=np.max(cm))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)

    import matplotlib.pyplot as plt  # 导入作图库
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.matshow(cm, cmap=plt.cm.Blues)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar(sm)  # 颜色标签

    for m in range(len(cm)):  # 数据标签
        for n in range(len(cm)):
            plt.annotate(cm[m, n], xy=(n, m), horizontalalignment='center', verticalalignment='center')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    l = len(np.unique(pd.concat([pd.DataFrame(y), pd.DataFrame(yp)], axis=0)))
    ax.set_xticks(range(0, l+1, 1))
    ax.set_yticks(range(0, l+1, 1))
    plt.xlim(-0.5, l - 0.5)
    plt.ylim(l - 0.5, -0.5)
    ax.set_xticklabels(np.unique(pd.concat([pd.DataFrame(y), pd.DataFrame(yp)], axis=0)))
    ax.set_yticklabels(np.unique(pd.concat([pd.DataFrame(y), pd.DataFrame(yp)], axis=0)))
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.show()