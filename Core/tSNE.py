import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import CDataset

#=====================================================================
def tSNEPlots(strOutputFile, df, listLabels):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    dfLocal = df.drop('Labels', axis='columns').copy()
    print(dfLocal.shape)
    tsne_results = tsne.fit_transform(df)

    df['Feature-one'] = tsne_results[:,0]
    df['Feature-two'] = tsne_results[:,1]

    df['Labels'] = df['Labels'].replace({0: listLabels[0]})
    df['Labels'] = df['Labels'].replace({1: listLabels[1]})

    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        x="Feature-one", y="Feature-two",
        hue="Labels",
        style="Labels",
        palette=['blue', 'red'],
        data=df,
        legend="full",
        #alpha=0.3
    )
    plt.legend(loc='upper right')
    plt.grid(linestyle='dotted')
    plt.savefig(strOutputFile, dpi=300, bbox_inches='tight')
    plt.show()

# DIGIT
def Digit(bVerbose = False):
    listData, listLabel = objDS.PrepareMNISTDS()
    print(listData.shape, listLabel.reshape(-1, 1).shape)
    # print(listLabels)
    
    if bVerbose:
        fTestSize = 0.7
        X_trainALL, X_test, y_trainALL, y_test = train_test_split(
                listData, listLabel.ravel(), 
                test_size=fTestSize,  random_state = 42)
        print('Total:' , listData.shape, 'Training:', X_trainALL.shape)

        unique, counts = np.unique(y_trainALL, return_counts=True)
        print(unique, counts)

        unique, counts = np.unique(y_test, return_counts=True)
        print(unique, counts)

        listDS = np.hstack((X_trainALL, y_trainALL.reshape(-1, 1)))
        print(listDS.shape)
    else:
        listDS = np.hstack((listData, listLabel.reshape(-1, 1)))
        print(listDS.shape)
        
    dfCols = ['Bin_'+str(i) for i in range(listData.shape[1])]
    dfCols.append('Labels')
    #print(dfCols)
    df = pd.DataFrame(listDS, columns=dfCols)
    return df

# KYOTO
def Kyoto(strFileName, bVerbose = False):
    #objDS.CreateBalanceDS(strFileName)

    listData, listLabel = objDS.GetKyotoDataset(strFileName)
    print('Kyoto: ', listData.shape, listLabel.shape)

    if bVerbose:
        fTestSize = 0.9
        X_trainALL, X_test, y_trainALL, y_test = train_test_split(
                listData, listLabel.ravel(), 
                test_size=fTestSize,  random_state = 42)
        print('Total:' , listData.shape, 'Training:', X_trainALL.shape)

        unique, counts = np.unique(y_trainALL, return_counts=True)
        print(unique, counts)

        unique, counts = np.unique(y_test, return_counts=True)
        print(unique, counts)
        
        listDS = np.hstack((X_trainALL, y_trainALL.reshape(-1, 1)))
        print(listDS.shape)
    else:
        listDS = np.hstack((listData, listLabel))
        print(listDS.shape)

    # Here add header to the dataset and also add labels 
    dfCols = ['Bin_'+str(i) for i in range(listData.shape[1])]
    dfCols.append('Labels')
    #print(dfCols)
    df = pd.DataFrame(listDS, columns=dfCols)
    return df

# BETH
def Beth(strFileName, nExp=1):
    listData, listLabel = objDS.GetBethDataset(strFileName)
    print(listData.shape, listLabel.shape)

    if nExp == 0:
        listDS = np.hstack((listData, listLabel))
        print(listDS.shape)
    elif nExp == 1:
        fTestSize = 0.9
        X_trainALL, X_test, y_trainALL, y_test = train_test_split(
                listData, listLabel.ravel(), 
                test_size=fTestSize,  random_state = 42)
        print('Total:' , listData.shape, 
              '\nTraining:', X_trainALL.shape)
        unique, counts = np.unique(y_trainALL, return_counts=True)
        print(unique, counts)

        listDS = np.hstack((X_trainALL, y_trainALL.reshape(-1, 1)))
        print(listDS.shape)
    elif nExp == 2:
        print('Nothing')

    dfCols = ['Bin_'+str(i) for i in range(listData.shape[1])]
    dfCols.append('Labels')
    #print(dfCols)
    df = pd.DataFrame(listDS, columns=dfCols)
    return df


################################################################
if __name__ == '__main__':
    print(os.getcwd())

    objDS = CDataset()

    cDATA = '' #'Kyoto'

    if cDATA == 'Kyoto':
        strFileName = r'./DATA/Kyoto2015DS.csv'
        strOutputFile = './Results25Aug23/tsneKyoto_train.pdf'
        df = Kyoto(strFileName, True)
    elif cDATA == 'Beth':
        strFileName = r'./DATA/Beth_16Aug2023.csv'
        strOutputFile = './Results25Aug23/tsneBeth_train.pdf'
        df = Beth(strFileName, nExp=1)
        df = df.sample(frac=0.1)
        print('Sampled: ', df['Labels'].value_counts())
    else:
        strOutputFile = './Results25Aug23/tsneDigit_train.pdf'
        df = Digit(bVerbose=True)

    tSNEPlots(strOutputFile, df, listLabels=['Positive', 'Negative'])
