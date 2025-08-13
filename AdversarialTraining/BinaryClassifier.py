#https://scikit-learn.org/0.15/modules/model_evaluation.html

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

#from Utils.tsnePlot import CTSNEPlots

#========================================================
# https://scikit-learn.org/stable/
class CClassification:
    def __init__(self):
        print('CClassification Object Created')

    def ReadFile(self, strFileName):
        print(strFileName)
        arrData = np.genfromtxt(strFileName, delimiter=',')  # from csv to numeric matrix
        return arrData
    
    def combineFiles(self, listA, listB, listC):
        return np.hstack((listA, listB, listC))
    
    def SetDataset(self, listData, listLabel):
        self.mArrDataSet = listData
        self.mArrLabels = listLabel
        return self.mArrDataSet, self.mArrLabels
    
    def CreateDataSet(self, arrDataBenign, arrDataMalacious, nObsPerClass = -1):
        nBenignSamples = len(arrDataBenign)
        nMaliciousSamples = len(arrDataMalacious)

        if nObsPerClass == -1:
            self.mArrDataSet = np.vstack((arrDataBenign, arrDataMalacious))
        else:
            nObsPerClass = min(nBenignSamples, nMaliciousSamples)
            # concatenates the data with equal size
            self.mArrDataSet = np.vstack((arrDataBenign[0:nObsPerClass,:], 
                                    arrDataMalacious[0:nObsPerClass,:])) 
            nBenignSamples = nObsPerClass
            nMaliciousSamples = nObsPerClass
        
        print('Len Benign = ', nBenignSamples,
              'Len Malicious = ', nMaliciousSamples)

        # Benign Label
        arrLabelsBenign = np.ones(nBenignSamples, dtype=int) * 1
        arrLabelsBenign = arrLabelsBenign.reshape(nBenignSamples, 1)
        #print(np.shape(arrLabelsBenign))

        # Malicious Label
        arrLabelsMalicious = np.ones(nMaliciousSamples, dtype=int) * -1
        arrLabelsMalicious = arrLabelsMalicious.reshape(nMaliciousSamples, 1)
        #print(np.shape(arrLabelsMalicious))

        # concatenates the labels
        self.mArrLabels = np.vstack((arrLabelsBenign, arrLabelsMalicious))  
        return self.mArrDataSet, self.mArrLabels
    
    def SelectClassifier(self, strToken = 'RF'):
        self.mStrModel = strToken

        if self.mStrModel == 'SVC':
            self.mModel = SVC(probability=True)
        elif self.mStrModel == 'SVC2':
            self.mModel = SVC(kernel='rbf', C=1e9, 
                              gamma=1e-07, probability=True) # kernel='linear', C=1 #C=1000,gamma = 1, kernel='linear'
        elif self.mStrModel == 'RF':
            self.mModel = RandomForestClassifier() # n_estimators=100, max_depth=10, random_state=42
        elif self.mStrModel == 'DT':
            self.mModel = DecisionTreeClassifier(probability=True) # max_depth=2
        elif self.mStrModel == 'MLP':
            self.mModel = MLPClassifier()
        elif self.mStrModel == 'NB':
            self.mModel = GaussianNB(probability=True)
        elif self.mStrModel == 'KNN':
            self.mModel = KNeighborsClassifier(probability=True) #n_neighbors=3

    def ShowResult(self, Y, y_true, y_pred):
        print('Accuracy=', self.mModel.score(Y, y_true))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        fPrecision = round(100*tp/(tp+fp), 2)
        fRecall = round(100*tp/(tp+fn), 2)

        F1_score = f1_score(y_true, y_pred, average='weighted')
        #F1_score = 2 * (fPrecision * fRecall) / (fPrecision + fRecall)
        F1_score = round(100 * F1_score, 2)
        
        # model accuracy for DataTest - 
        # Return the mean accuracy on the given test data and labels.
        fAccuracy =  self.mModel.score(Y, y_true)
        fAccuracy = round(100 * fAccuracy, 2)

        print('Model:', self.mStrModel, 'A:', str(fAccuracy), 
            ' P:', str(fPrecision), ' R:' + str(fRecall), ' F1:', F1_score)

    def Execute(self, fTrainSz, strModelName, strIter,
                bPlot=False, bPredict=False):
        X, Y, x_true, y_true = train_test_split(self.mArrDataSet, 
                                                self.mArrLabels.ravel(),
                                                stratify=self.mArrLabels,
                                                train_size=fTrainSz,
                                                random_state=0)
        print('Training Data Size:', X.shape)
        
        self.mModel.fit(X, x_true)
        # save the model to disk
        pickle.dump(self.mModel, open(strModelName, 'wb'))

        y_pred = self.mModel.predict(Y)
        self.ShowResult(Y, y_true, y_pred)

        # load the model from disk
        Loaded_model = pickle.load(open(strModelName, 'rb'))
        result = Loaded_model.score(Y, y_true)
        print('Loaded model accuracy: ', result*100)

        if bPredict:
            target_names = ['Original', 'Synthetic']
            print(classification_report(y_true, y_pred, target_names=target_names))

            list_y_probs = self.mModel.predict_proba(Y)

            print(y_pred.shape, list_y_probs.shape)

            for n in range(y_true.shape[0]):
                print(y_true[n], y_pred[n], list_y_probs[n][y_true[n]])
                break

            print(list_y_probs[y_true])
        
        if bPlot:
            strName = 'SVM: Orig vs Syn' + strIter
            # PrecisionRecallDisplay
            display = PrecisionRecallDisplay.from_estimator(
                self.mModel, Y, y_true, name=strName, ax=ax)
            #_ = display.ax_.set_title("Two-class Precision-Recall curve")
            strOut = '../Images/PR' + strIter + '_' + str(fTrainSz) + '.pdf'
            #plt.savefig(strOut, dpi=300)
            #plt.show()

        #return fAccuracy, fPrecision, fRecall, F1_score
        # END Execute
    
    def PlotHyperplane(self, fTrainSz):
        X, Y, x_true, y_true = train_test_split(self.mArrDataSet, 
                                                self.mArrLabels.ravel(),
                                                stratify=self.mArrLabels,
                                                train_size=fTrainSz,
                                                random_state=0)
        
        # fit the model and get the separating hyperplane
        clf = SVC(kernel="linear", C=1.0)
        clf.fit(X, x_true)

        # fit the model and get the separating hyperplane using weighted classes
        wclf = SVC(kernel="linear", class_weight={1: 10})
        wclf.fit(X, x_true)

        # plot the samples
        plt.scatter(X[:, 0], X[:, 1], c=x_true, cmap=plt.cm.Paired, edgecolors="k")

        # plot the decision functions for both classifiers
        ax = plt.gca()
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            plot_method="contour",
            colors="k",
            levels=[0],
            alpha=0.5,
            linestyles=["-"],
            ax=ax,
        )

        # plot decision boundary and margins for weighted classes
        wdisp = DecisionBoundaryDisplay.from_estimator(
            wclf,
            X,
            plot_method="contour",
            colors="r",
            levels=[0],
            alpha=0.5,
            linestyles=["-"],
            ax=ax,
        )

        plt.legend(
            [disp.surface_.collections[0], wdisp.surface_.collections[0]],
            ["non weighted", "weighted"],
            loc="upper right",
        )
        plt.show()

#########################################################    
def Example():
    # we create two clusters of random points
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [1.5, 0.5]
    X, y = make_blobs(
        n_samples=[n_samples_1, n_samples_2],
        centers=centers,
        cluster_std=clusters_std,
        random_state=0,
        shuffle=False,
    )

    print(X.shape, y.shape)

#========================================================
if __name__ == '__main__':
    print('Root Path:', os.getcwd())