import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


class PCA:

    def fit(x_train,y_train):
        # y_train = y_train['Type'].tolist()
        #Normalization
        x_train = (x_train - x_train.mean()) / x_train.std(ddof=0)

        df_corr = (1 / x_train.shape[0]) * x_train.T.dot(x_train)

        # plt.figure(figsize=(10, 10))
        # sns.heatmap(df_corr, vmax=1, square=True, annot=True)
        # plt.title('Correlation matrix')
        # plt.show()

        u,s,v = np.linalg.svd(df_corr)
        eig_values, eig_vectors = s, u

        # first seven eigen vectors is enough
        x=0;
        for i in range(2):
             x = x+eig_values[i]
             # print(eig_values[i])

        print("Explained varience for "+str(i+1) + " = ", x/sum(eig_values))
        print("Explained varience for "+str(i+2)+ " = ", (x+eig_values[2])/sum(eig_values))


        # Finding the principal components
        pc1 = x_train.dot(eig_vectors[:,0])
        pc2 = x_train.dot(eig_vectors[:,1])
        pc3 = x_train.dot(eig_vectors[:,2])

        col= ['PC1','PC2','PC3']
        global pcs
        pcs = pd.concat([pc1, pc2,pc3], axis=1)
        pcs.columns = col
        pcs['label'] = y_train
        return pcs

    def plot_2D():

        # result = pd.DataFrame(pc1, columns=['PC1'])
        # result['PC2'] = pc2

        # print(result)
        fig2=sns.lmplot('PC1', 'PC2', data= pcs, fit_reg=False,  # x-axis, y-axis, data, no line
                   scatter_kws={"s": 30},height=6, aspect=1, hue="label",legend=False) # color

        # title
        plt.legend(bbox_to_anchor=(1, 1), loc=2)

        # plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
        plt.title('PCA result')
        plt.tight_layout()
        plt.show()
        fig2.savefig('pca2d.png', bbox_inches='tight')

    def plot_3D():
        fig = px.scatter_3d(pcs, x='PC1', y='PC2', z='PC3',color= 'label')
        fig.show()

