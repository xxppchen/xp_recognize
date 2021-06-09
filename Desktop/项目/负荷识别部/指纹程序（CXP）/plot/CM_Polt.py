def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数

    cm = confusion_matrix(y, yp)  # 混淆矩阵

    import matplotlib.pyplot as plt  # 导入作图库
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.matshow(cm, cmap=plt.cm.Blues)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    # plt.colorbar()  # 颜色标签

    for m in range(len(cm)):  # 数据标签
        for n in range(len(cm)):
            plt.annotate(cm[m, n], xy=(n, m), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.show()