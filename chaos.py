from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

def logistic_regression_boundary(data, idx, title:str, show_plot=False, C=1, multi_class="multinomal", max_iter=1000):
    X= np.array(data.drop(['y', 'cluster_id'], axis=1))
    y= np.array(data["y"], dtype=int)

    logistic_reg = LogisticRegression(C=1, multi_class='multinomial', max_iter=1000)
    logistic_reg.fit(X[idx,:], y[idx])
    if show_plot:
        plot_decision_regions(X,y, clf=logistic_reg, scatter_kwargs={'s':0})
        sns.scatterplot(data=X[idx,:], x=X[idx,0], y=X[idx,1], color="black", marker="+", s=150)
        plt.title(title)
        plt.show()

    return logistic_reg.score(X[idx,:], y[idx]), logistic_reg.score(X,y)

