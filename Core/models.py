
import os
import sys


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from scipy import interpolate

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split

from .dataset import CDataset
from .plots import CPlots

######################################################################################
# Class CModels
class CModels:
    def __init__(self, strDirPath, fTestSize, nclfA, nclfD, nclfR):
        print('CModels Object Created')
        self.m_strDirPath = strDirPath
        self.m_nclfA = nclfA
        self.m_nclfD = nclfD
        self.m_nclfR = nclfR
        self.mfTestSize = fTestSize #0.7

    def chooseClassifier(self, nclf):
        clfs = [KNeighborsClassifier(5),  # 0
                DecisionTreeClassifier(max_depth=5),  # 1
                RandomForestClassifier(),  # 2
                RandomForestClassifier(
                    max_depth=5, n_estimators=10, max_features=5),  # 3
                MLPClassifier(alpha=1, max_iter=100),  # 4
                AdaBoostClassifier(),  # 5
                GaussianNB()]  # 6
        return (clfs[nclf])

    # ----------------------------------------------------------------------
    def PlotROC(self, strFileName):
        objPlots = CPlots(self.nTestPos, self.nTestNeg)

        fig = plt.figure()  # plot roc curves
        objPlots.plot_tpr(self.predD, self.y_test, self.clfD_name)
        objPlots.plot_tpr(self.predR, self.y_test, self.clfR_name)
        objPlots.plot_tpr(self.predA, self.y_test, self.clfA_name)

        plt.legend()
        plt.grid(True)
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rates (TPR)', fontsize=12)
        plt.title('ROC curves: True Positive Rate (TPR) for clfD, clfA, clfR')

        plt.savefig(strFileName, dpi=300, bbox_inches='tight')

        plt.show()

    def RunROC(self, listData, listLabel):
        # Split the dataset
        X_train, X_test, y_train, self.y_test = train_test_split(listData,
                                                                 listLabel.ravel(),
                                                                 test_size=0.5,
                                                                 shuffle=False,
                                                                 stratify=None)

        print('Training Dataset', X_train.shape)
        print('Testing Dataset', X_test.shape)

        clfA = self.chooseClassifier(self.m_nclfA)
        clfD = self.chooseClassifier(self.m_nclfD)
        clfR = self.chooseClassifier(self.m_nclfR)

        self.clfD_name = 'clfD: ' + str(type(clfD)).split(".")[-1][:-2]
        self.clfA_name = 'clfA: ' + str(type(clfA)).split(".")[-1][:-2]
        self.clfR_name = 'clfR: ' + str(type(clfR)).split(".")[-1][:-2]

        clfD.fit(X_train, y_train)
        clfA.fit(X_train, y_train)
        clfR.fit(X_train, y_train)

        # number of pos & neg in test
        self.nTestPos = self.y_test.sum()
        self.nTestNeg = len(self.y_test) - self.nTestPos

        # define predD, predA, predR
        self.predD = clfD.predict_proba(X_test)[:, 1]  # defender 1 predictions
        self.predA = clfA.predict_proba(
            X_test)[:, 1]  # adversary's predictions
        self.predR = clfR.predict_proba(X_test)[:, 1]  # defender 2 predictions

    # ---------------------------------------------------------------------------
    # define roc_curve and adversarial roc curve functions

    def PlotAROC(self, strFileName):
        objPlots = CPlots(self.nTestPos, self.nTestNeg)

        # plot roc curves
        fig = plt.figure(1)
        objPlots.plot_tpr(self.predD, self.y_test, self.clfD_name)
        objPlots.plot_tpr(self.predA, self.y_test, self.clfA_name)
        plt.legend()
        plt.grid(True)
        plt.show()

        # aroc_curve
        fpD, kfnD, afnD = objPlots.aroc_curve(
            self.y_test, self.predD, self.predA)

        # plot adversarial roc curves
        fig = plt.figure(2)
        objPlots.plot_tpr(self.predA, self.y_test, self.clfA_name)
        objPlots.plot_tpr(self.predD, self.y_test, self.clfD_name)

        utpr1 = objPlots.plot_utpr(fpD, kfnD, self.clfD_name)
        afr1 = objPlots.plot_afr(fpD, kfnD, afnD, self.clfD_name)

        plt.legend()
        plt.title('ROC, utpr ROC and adversarial failure curves')
        plt.show()

    def RunAROC(self, listData, listLabel):
        # Split the dataset
        X_train, X_test, y_train, self.y_test = train_test_split(listData,
                                                                 listLabel.ravel(),
                                                                 test_size=0.5,
                                                                 shuffle=False,
                                                                 stratify=None)

        print('Training Dataset', X_train.shape)
        print('Testing Dataset', X_test.shape)

        clfA = self.chooseClassifier(self.m_nclfA)
        clfD = self.chooseClassifier(self.m_nclfD)

        self.clfD_name = 'clfD: '+str(type(clfD)).split(".")[-1][:-2]
        self.clfA_name = 'clfA: '+str(type(clfA)).split(".")[-1][:-2]

        # fit & predict
        clfD.fit(X_train, y_train)
        clfA.fit(X_train, y_train)

        self.nTestPos = self.y_test.sum()
        self.nTestNeg = len(self.y_test) - self.nTestPos

        # define predD, predA, predR
        self.predD = clfD.predict_proba(X_test)[:, 1]  # defender's  predictions
        self.predA = clfA.predict_proba(X_test)[:, 1]  # adversary's predictions

    # ---------------------------------------------------------------------------
    # Example 3
    # prepare training & test sets, fit clfD and clfA, return predictions predA/D/R
    def prepDataPreds(self):
        # Split data into train and test subsets, fit clfD and clfA
        X_train, X_test, y_train, y_test = train_test_split(
            self.mlistData, self.mlistLabel.ravel(),
            test_size=0.3, shuffle=False)

        testp = y_test.sum()
        testn = len(y_test)-testp  # n. of pos & neg in test

        self.clfD.fit(X_train, y_train)
        self.clfA.fit(X_train, y_train)

        # define predD, predA, predR (defender, adversary, randomized)
        predD = self.clfD.predict_proba(X_test)[:, 1]  # defender's predictions
        predA = self.clfA.predict_proba(X_test)[:, 1]  # adversary's predictions
        rnums = np.random.randint(100, size=len(predA))/100

        # randomized defender (predA+random)
        predR = predA * (1-self.rw100/100) + rnums*self.rw100/100
        return (predD, predA, predR, y_test, testp, testn)

    def Run_UTPR_AFR(self, listData, listLabel):
        self.mlistData = listData
        self.mlistLabel = listLabel

        self.rw100 = 30  # randomization rate for clfR = rw100/100

        self.clfD = RandomForestClassifier(
            max_depth=8, n_estimators=10, max_features=3)
        self.clfA = RandomForestClassifier(
            max_depth=8, n_estimators=10, max_features=3)

        self.clfD_name = 'clfD: ' + \
            str(type(self.clfD)).split(".")[-1][:-2]  # keyed Defender
        self.clfA_name = 'clfA: ' + \
            str(type(self.clfA)).split(".")[-1][:-2]  # Adversary
        self.clfR_name = 'clfR: ' + \
            str(self.rw100) + "%"+' random + clfA'    # Randomized defender

        # just to find the length
        predD, predA, predR, y_test, testp, testn = self.prepDataPreds()
        objPlots_Local = CPlots(testp, testn)

        lenD = len(objPlots_Local.getThreshold(predD))
        lenR = len(objPlots_Local.getThreshold(predR))
        fprDs = np.zeros(lenD)
        tprDs = np.zeros(lenD)

        roc_aucDs = 0  # fpr/tpr values for D
        fprRs = np.zeros(lenR)
        tprRs = np.zeros(lenR)

        roc_aucRs = 0  # fpr/tpr values for R
        fprDus = np.zeros(lenD)
        utprDs = np.zeros(lenD)

        aroc_aucDus = 0  # fpr/utpr values for D
        fprRus = np.zeros(lenR)
        utprRs = np.zeros(lenR)

        aroc_aucRus = 0  # fpr/utpr values for R
        fprDas = np.zeros(lenD)
        afrDs = np.zeros(lenD)

        aroc_aucDas = 0  # fpr/afr values for D
        fprRas = np.zeros(lenD)
        afrRs = np.zeros(lenD)

        aroc_aucRas = 0  # fpr/afr values for R
        # number of iterations (repeat and plot mean tpr/utpr/afr curves)
        nRepeats = 10
        for i in range(nRepeats):  # repeat everything below 'repeats' times, acc results
            print('Iteration:', i)
            # split dataset, get Preds
            predD, predA, predR, y_test, testp, testn = self.prepDataPreds()  
            objPlots = CPlots(testp, testn)

            # tpr with mean computation
            fprD, tprD, roc_aucD = objPlots.compute_tprAUROC(predD, y_test)
            fprR, tprR, roc_aucR = objPlots.compute_tprAUROC(predR, y_test)
            maxlenD = max(len(fprD), len(fprDs))
            if len(fprDs) < maxlenD:
                f1 = interpolate.interp1d(np.arange(0, len(fprDs)), fprDs)
                fprDs = f1(np.linspace(0.0, len(fprDs)-1, len(fprD)))
                f2 = interpolate.interp1d(np.arange(0, len(tprDs)), tprDs)
                tprDs = f2(np.linspace(0.0, len(tprDs)-1, len(tprD)))
            else:
                f1 = interpolate.interp1d(np.arange(0, len(fprD)), fprD)
                fprD = f1(np.linspace(0.0, len(fprD)-1, len(fprDs)))
                f2 = interpolate.interp1d(np.arange(0, len(tprD)), tprD)
                tprD = f2(np.linspace(0.0, len(tprD)-1, len(tprDs)))
            fprDs += fprD
            tprDs += tprD
            roc_aucDs += roc_aucD  # sum up
            maxlenR = max(len(fprR), len(fprRs))
            if len(fprRs) < maxlenR:
                f1 = interpolate.interp1d(np.arange(0, len(fprRs)), fprRs)
                fprRs = f1(np.linspace(0.0, len(fprRs)-1, len(fprR)))
                f2 = interpolate.interp1d(np.arange(0, len(tprRs)), tprRs)
                tprRs = f2(np.linspace(0.0, len(tprRs)-1, len(tprR)))
            else:
                f1 = interpolate.interp1d(np.arange(0, len(fprR)), fprR)
                fprR = f1(np.linspace(0.0, len(fprR)-1, len(fprRs)))
                f2 = interpolate.interp1d(np.arange(0, len(tprR)), tprR)
                tprR = f2(np.linspace(0.0, len(tprR)-1, len(tprRs)))
            fprRs += fprR
            tprRs += tprR
            roc_aucRs += roc_aucR  # sum up

            # utpr with mean computation
            fpD, kfnD, afnD = objPlots.aroc_curve(y_test, predD, predA)
            fpR, kfnR, afnR = objPlots.aroc_curve(y_test, predR, predA)
            fprDu, utprD, aroc_aucDu = objPlots.compute_utprAUROC(fpD, kfnD)
            fprRu, utprR, aroc_aucRu = objPlots.compute_utprAUROC(fpR, kfnR)
            maxlenD = max(len(fprDu), len(fprDus))
            if len(fprDus) < maxlenD:
                f1 = interpolate.interp1d(np.arange(0, len(fprDus)), fprDus)
                fprDus = f1(np.linspace(0.0, len(fprDus)-1, len(fprDu)))
                f2 = interpolate.interp1d(np.arange(0, len(utprDs)), utprDs)
                utprDs = f2(np.linspace(0.0, len(utprDs)-1, len(utprD)))
            else:
                f1 = interpolate.interp1d(np.arange(0, len(fprDu)), fprDu)
                fprDu = f1(np.linspace(0.0, len(fprDu)-1, len(fprDus)))
                f2 = interpolate.interp1d(np.arange(0, len(utprD)), utprD)
                utprD = f2(np.linspace(0.0, len(utprD)-1, len(utprDs)))
            fprDus += fprDu
            utprDs += utprD
            aroc_aucDus += aroc_aucDu  # sum up
            maxlenR = max(len(fprRu), len(fprRus))
            if len(fprRus) < maxlenR:
                f1 = interpolate.interp1d(np.arange(0, len(fprRus)), fprRus)
                fprRus = f1(np.linspace(0.0, len(fprRus)-1, len(fprRu)))
                f2 = interpolate.interp1d(np.arange(0, len(utprRs)), utprRs)
                utprRs = f2(np.linspace(0.0, len(utprRs)-1, len(utprR)))
            else:
                f1 = interpolate.interp1d(np.arange(0, len(fprRu)), fprRu)
                fprRu = f1(np.linspace(0.0, len(fprRu)-1, len(fprRus)))
                f2 = interpolate.interp1d(np.arange(0, len(utprR)), utprR)
                utprR = f2(np.linspace(0.0, len(utprR)-1, len(utprRs)))
            fprRus += fprRu
            utprRs += utprR
            aroc_aucRus += aroc_aucRu  # sum up

            # afr with mean computation (same as for utpr above)
            fprDa, afrD, aroc_aucDa = objPlots.compute_afrAUROC(
                fpD, kfnD, afnD)
            fprRa, afrR, aroc_aucRa = objPlots.compute_afrAUROC(
                fpR, kfnR, afnR)
            maxlenD = max(len(fprDa), len(fprDas))
            if len(fprDas) < maxlenD:
                f1 = interpolate.interp1d(np.arange(0, len(fprDas)), fprDas)
                fprDas = f1(np.linspace(0.0, len(fprDas)-1, len(fprDa)))
                f2 = interpolate.interp1d(np.arange(0, len(afrDs)), afrDs)
                afrDs = f2(np.linspace(0.0, len(afrDs)-1, len(afrD)))
            else:
                f1 = interpolate.interp1d(np.arange(0, len(fprDa)), fprDa)
                fprDa = f1(np.linspace(0.0, len(fprDa)-1, len(fprDas)))
                f2 = interpolate.interp1d(np.arange(0, len(afrD)), afrD)
                afrD = f2(np.linspace(0.0, len(afrD)-1, len(afrDs)))
            fprDas += fprDa
            afrDs += afrD
            aroc_aucDas += aroc_aucDa  # sum up
            maxlenR = max(len(fprRa), len(fprRas))
            if len(fprRas) < maxlenR:
                f1 = interpolate.interp1d(np.arange(0, len(fprRas)), fprRas)
                fprRas = f1(np.linspace(0.0, len(fprRas)-1, len(fprRa)))
                f2 = interpolate.interp1d(np.arange(0, len(afrRs)), afrRs)
                afrRs = f2(np.linspace(0.0, len(afrRs)-1, len(afrR)))
            else:
                f1 = interpolate.interp1d(np.arange(0, len(fprRa)), fprRa)
                fprRa = f1(np.linspace(0.0, len(fprRa)-1, len(fprRas)))
                f2 = interpolate.interp1d(np.arange(0, len(afrR)), afrR)
                afrR = f2(np.linspace(0.0, len(afrR)-1, len(afrRs)))
            fprRas += fprRa
            afrRs += afrR
            aroc_aucRas += aroc_aucRa  # sum up
        # END FOR

        # compute mean tpr,utpr,afr curves, and corresponding AUCs
        # 1) mean tpr for clfD and corresponding roc_AUC
        mfprD = fprDs/nRepeats
        mtprD = tprDs/nRepeats
        mroc_aucD = roc_aucDs/nRepeats  # tpr

        # 2) mean tpr for clfR and corresponding roc_AUC
        mfprR = fprRs/nRepeats
        mtprR = tprRs/nRepeats
        mroc_aucR = roc_aucRs/nRepeats  # tpr

        # 3) mean utpr and afr for clfD w.r.t. clfA and corresponding utpr/afr_roc_AUC
        mfprDu = fprDus/nRepeats
        mutprD = utprDs/nRepeats
        maroc_aucDu = aroc_aucDus/nRepeats  # utpr
        mfprDa = fprDas/nRepeats
        mafrD = afrDs/nRepeats
        maroc_aucDa = aroc_aucDas/nRepeats  # afr

        # 4) mean utpr and afr for clfR w.r.t. clfA and corresponding utpr/afr_roc_AUC
        mfprRu = fprRus/nRepeats
        mutprR = utprRs/nRepeats
        maroc_aucRu = aroc_aucRus/nRepeats  # utpr
        mfprRa = fprRas/nRepeats
        mafrR = afrRs/nRepeats
        maroc_aucRa = aroc_aucRas/nRepeats  # afr

        objPlots = CPlots(-1, -1)
        fig = plt.figure(1)
        objPlots.arocPlotCol(mfprD, mtprD, np.round(
            mroc_aucD, 2), self.clfD_name, '#FF6C02')
        objPlots.arocPlotCol(mfprR, mtprR, np.round(
            mroc_aucR, 2), self.clfR_name, 'green')
        plt.title('mean tpr for clfD and clfR (ROC curves)')
        plt.legend()
        plt.show()

        # plot mean tpr, utpr and afr for clfD w.r.t. clfA
        fig = plt.figure(2)
        objPlots.arocPlotCol(mfprD, mtprD, np.round(
            mroc_aucD, 2), "tpr ", '#FF6C02')
        objPlots.arocPlotCol(mfprDu, mutprD, np.round(
            maroc_aucDu, 2), "utpr ", 'green')
        objPlots.arocPlotCol(mfprDa, mafrD, np.round(
            maroc_aucDa, 2), "afr ", 'red')
        plt.title('mean tpr/utpr/afr for ' + self.clfD_name +
                  '\n w.r.t. ' + self.clfA_name)
        plt.legend()
        plt.show()

        # plot behaviours of clfD and clfR in terms of utpr, w.r.t. clfA
        fig = plt.figure(3)
        objPlots.arocPlot(mfprDu, mutprD, np.round(
            maroc_aucDu, 2), self.clfD_name)
        objPlots.arocPlot(mfprRu, mutprR, np.round(
            maroc_aucRu, 2), self.clfR_name)
        plt.title(
            'mean unknown True Positive Rate (utpr) for \n clfD,clfR w.r.t. ' + self.clfA_name)
        plt.legend()
        plt.show()

        # plot behaviours of clfD and clfR in terms of afr w.r.t. clfA
        fig = plt.figure(4)
        # same plot function as above
        objPlots.arocPlot(mfprDa, mafrD, np.round(
            maroc_aucDa, 2), self.clfD_name)
        objPlots.arocPlot(mfprRa, mafrR, np.round(
            maroc_aucRa, 2), self.clfR_name)  # just use the utpr plot
        plt.title(
            'mean Adversarial Failure Rate (afr) for \n clfD,clfR w.r.t. ' + self.clfA_name)
        plt.legend()
        plt.show()

    # ------------------------------------------------------------------------
    # threshold classification & roc analysis
    # prepare random trainset subset
    def tsetPrep(self, trainSize, X_trainALL, y_trainALL):
        # repeat to find a subset with 2 classes
        for i in range(20):
            idx = np.random.choice(np.arange(len(X_trainALL)),
                                   round(len(X_trainALL)*trainSize), replace=False)
            # subset of X_trainaALL of size trainSize
            SX_train = X_trainALL[idx]
            Sy_train = y_trainALL[idx]  # corresponding classifications
            if np.any(Sy_train) and np.any(Sy_train-1):  # not all 0s nor all 1s
                return (SX_train, Sy_train)  # subset acceptable, return it
            else:
                print('train set too small')  # try again
        # never found a good one, will crash
        print('train set always too small')
        exit()
        
    # compute AUC wrt tpr, udpr and afr for R, and wrt to tpr for A
    # fit clfA, compute predictions predA, randomize into predR
    # Split data into train and test subsets, fit clfA ###############
    def predictRA(self, bBeth):
        trainSize = 1.0
        #testSize = 0.7  # size of test set out of the whole data set

        if bBeth == False:
            X_trainALL, X_test, y_trainALL, y_test = train_test_split(
                self.mlistData, self.mlistLabel.ravel(), 
                test_size=self.mfTestSize, random_state = 42) #, shuffle=False)
        else:
            X_trainALL, X_test, y_trainALL, y_test = train_test_split(
                self.mlistData, self.mlistLabel.ravel(), 
                test_size=self.mfTestSize, shuffle=False)
        
        XD_train, yD_train = self.tsetPrep(trainSize, X_trainALL, y_trainALL)
        self.clfA.fit(XD_train, yD_train)

        # compute preditction list predA (for each element of X_test)
        predA = self.clfA.predict_proba(X_test)[:, 1]  # adversary's predictions

        # compute a list of random numbers between 0 and 100, of length len(predA)
        rnums = np.random.randint(100, size=len(predA))/100
        predR = predA*(1-self.rw100/100) + (rnums*self.rw100) / 100  # randomized predA
        return (predR, predA, y_test)

    def computeAUCsRA(self, bBeth=False):
        predR, predA, y_test = self.predictRA(bBeth)
        testp = y_test.sum()
        testn = len(y_test) - testp  # number of pos & neg in test

        objPlots = CPlots(testp, testn)

        # compute roc curves: AUC for tpr (for A and R)
        roc_aucR = objPlots.compute_tprAUROC_RV(predR, y_test)
        roc_aucA = objPlots.compute_tprAUROC_RV(predA, y_test)

        # compute adversarial roc curves for R wrt A (utpr and afr)
        fpR, kfnR, afnR = objPlots.aroc_curve(y_test, predR, predA)
        #utpr_aucR = objPlots.compute_utprAUROC_RV(fpR, kfnR)
        afr_aucR = objPlots.compute_afrAUROC_RV(fpR, kfnR, afnR)

        return (roc_aucR, roc_aucA, afr_aucR)

    def Run_RV(self, listData, listLabel, nRepeats, nSteps):
        self.mlistData = listData
        self.mlistLabel = listLabel

        self.clfA = self.chooseClassifier(self.m_nclfA)
        clfA_name = ' A: ' + str(type(self.clfA)).split(".")[-1][:-2]
        # clfR_name = ' R: random + ' + str(type(self.clfA)).split(".")[-1][:-2]

        # Repeat for different randomizations rw100 for predR
        # Initialize lists of mean values, will hold one for each element of range_rw100
        mean_tpr_aucAs = []
        std_tpr_aucAs = []

        mean_tpr_aucRs = []
        std_tpr_aucRs = []
        
        #mean_utpr_aucRs = []
        #std_utpr_aucRs = []
        
        mean_afr_aucRs = []
        std_afr_aucRs = []

        # new randoms at each run otherwise np.random.seed(0)
        np.random.seed(None)

        #nRepeats = 3  # number of iterations
        # size of steps from 0 to 100 for rw100 (0, step, 2*step, etc.)
        #nSteps = 5
        # [0, step, 2*step, ..., n] for n<=100
        arrTrainPercent = np.arange(0, 100+nSteps, nSteps)
        print(arrTrainPercent)

        # repeat for rw100 in [0,step,2*step, ...]
        for self.rw100 in arrTrainPercent:
            tpr_aucRs = []
            tpr_aucAs = []  # lists of tpr_AUC for R and A
            afr_aucRs = []
            #utpr_aucRs = []  # lists of afr_AUC and utpr_AUC for R wrt A

            for i in range(nRepeats):
                roc_aucR, roc_aucA, afr_aucR = self.computeAUCsRA()
                #utpr_aucRs += [utpr_aucR]
                afr_aucRs += [afr_aucR]  # accumulate results
                tpr_aucRs += [roc_aucR]
                tpr_aucAs += [roc_aucA]  # accumulate results

            print("rw100 = ", self.rw100, " repeats = ", nRepeats)
            print("np.mean(tpr_aucAs)", round(np.mean(tpr_aucAs), 3))
            print("np.mean(tpr_aucRs)", round(np.mean(tpr_aucRs), 3))
            #print("np.mean(utpr_aucRs)", round(np.mean(utpr_aucRs), 3))
            print("np.mean(afr_aucRs)", round(np.mean(afr_aucRs), 3))

            # add mean and std_dev of tpr,utpr & afr for R and tpr for A
            # to the corresponding lists, holding such values for each rw100 value
            mean_tpr_aucAs += [np.mean(tpr_aucAs)]
            std_tpr_aucAs += [np.std(tpr_aucAs)]

            mean_tpr_aucRs += [np.mean(tpr_aucRs)]
            std_tpr_aucRs += [np.std(tpr_aucRs)]

            #mean_utpr_aucRs += [np.mean(utpr_aucRs)]
            #std_utpr_aucRs += [np.std(utpr_aucRs)]

            mean_afr_aucRs += [np.mean(afr_aucRs)]
            std_afr_aucRs += [np.std(afr_aucRs)]
        #END FOR

        # SaveData
        arrColumns = ['train_percent', 
                      'mean_tpr_aucAs', 'std_tpr_aucAs', 
                      'mean_tpr_aucRs', 'std_tpr_aucRs',
                      'mean_afr_aucRs', 'std_afr_aucRs']
    
        df = pd.DataFrame(columns=arrColumns)
        df['train_percent'] = np.array(arrTrainPercent/100).tolist()

        df['mean_tpr_aucAs'] = np.array(mean_tpr_aucAs).tolist()
        df['std_tpr_aucAs'] = np.array(std_tpr_aucAs).tolist()

        df['mean_tpr_aucRs'] = np.array(mean_tpr_aucRs).tolist()
        df['std_tpr_aucRs'] = np.array(std_tpr_aucRs).tolist()
        
        df['mean_afr_aucRs'] = np.array(mean_afr_aucRs).tolist()
        df['std_afr_aucRs'] = np.array(std_afr_aucRs).tolist()

        filename = self.m_strDirPath + \
                    '/A-' + str(self.m_nclfA) + \
                    '_Steps-' + str(nSteps) + '_Rep-' + str(nRepeats)
        df.to_csv(filename + '.csv', index=False)

        # plot means for tpr (A and R), and for utpr and afr (R wrt A)
        fig, ax = plt.subplots()
        ax.plot(arrTrainPercent/100, mean_tpr_aucAs, label='tpr_AUC for clfA')
        ax.plot(arrTrainPercent/100, mean_tpr_aucRs, label='tpr_AUC for clfR')
        #ax.plot(arrTrainPercent/100, mean_utpr_aucRs, label='utpr_AUC for clfR')
        ax.plot(arrTrainPercent/100, mean_afr_aucRs, label='afr_AUC for clfR')
        ax.set_xticks(arrTrainPercent/100) 
        ax.legend(loc='best')

        plt.xticks(rotation ='vertical')
        plt.grid(True)
        plt.title(clfA_name + ", R: A+random in [0..1]")
        plt.savefig(filename + '_Rrep.png')
        #plt.show()
        plt.clf()

        # plot std_devs for utpr and afr (R wrt A)
        fig, ax = plt.subplots()

        #plt.errorbar(arrTrainPercent/100, mean_utpr_aucRs, std_utpr_aucRs, marker='^', label='utpr_AUC for clfR')
        ax.errorbar(arrTrainPercent/100, mean_afr_aucRs, std_afr_aucRs, marker='^', label='afr_AUC for clfR')
        ax.set_xticks(arrTrainPercent/100) 
        ax.legend(loc='best')
        #plt.legend(loc='lower right')

        plt.xticks(rotation ='vertical')
        plt.grid(True)
        plt.savefig(filename + '_meansR_std_repeat.png')
        #plt.show()
        plt.clf()

    # ------------------------------------------------------------------------
    # prepare D & A predicitons for xtest
    def predictDA(self, bBeth):  
        # Split data into train and test subsets (with testSize for test)
        if bBeth == False:
            X_trainALL, X_test, y_trainALL, y_test = train_test_split(
                self.mlistData, self.mlistLabel.ravel(), 
                test_size=self.mfTestSize, random_state = 42) #, shuffle=False)
        else:
            X_trainALL, X_test, y_trainALL, y_test = train_test_split(
                self.mlistData, self.mlistLabel.ravel(), 
                test_size=self.mfTestSize, shuffle=False)
        #print('predictDA::DATA', X_trainALL.shape, X_test.shape)
        #print('predictDA::LABEL:', np.bincount(y_trainALL), np.bincount(y_test))
        
        XD_train, yD_train = self.tsetPrep(self.mTrainSize, X_trainALL, y_trainALL.ravel())
        #print('predictDA::DATA_D', XD_train.shape, yD_train.shape)
        self.clfD.fit(XD_train, yD_train)

        XA_train, yA_train = self.tsetPrep(self.mTrainSize, X_trainALL, y_trainALL.ravel())
        #print('predictDA::DATA_A', XA_train.shape, yA_train.shape)
        # XA_train=X_trainALL; yA_train=y_trainALL
        self.clfA.fit(XA_train, yA_train)

        predD = self.clfD.predict_proba(X_test)[:, 1]  # defender's predictions
        predA = self.clfA.predict_proba(X_test)[:, 1]  # adversary's predictions

        return (predD, predA, y_test)

    # compute AUC wrt tpr, udpr and afr for D, and wrt tpr for A
    # RTRAIN  
    def computeAUCsDA(self, bBeth):  
        predD, predA, y_test = self.predictDA(bBeth)
        testp = y_test.sum()
        testn = len(y_test)-testp  # number of pos & neg in test

        objPlots = CPlots(testp, testn)

        # compute roc curves: AUC for tpr (for A and D)
        roc_aucD = objPlots.compute_tprAUROC_RV(predD, y_test)
        roc_aucA = objPlots.compute_tprAUROC_RV(predA, y_test)
        
        # compute adversarial roc curves for D wrt A (utpr and afr)
        fpD, kfnD, afnD = objPlots.aroc_curve(y_test, predD, predA)
        # utpr_aucD = objPlots.compute_utprAUROC_RV(fpD, kfnD)
        afr_aucD = objPlots.compute_afrAUROC_RV(fpD, kfnD, afnD)
        return (roc_aucD, roc_aucA, afr_aucD)
    
    # Deprecated - replaced by a csv file generation
    def SaveRTrainData(self, nStep, nRepeats, 
                 mean_tpr_aucDs, mean_utpr_aucDs,
                 mean_afr_aucDs, std_tpr_aucDs,
                 std_utpr_aucDs, std_afr_aucDs):
        # save results in text form to a file with meaningful name
        original_stdout = sys.stdout
        filename = '.\local-data\D:' + str(self.m_nclfD) + '_A:' + str(self.m_nclfA) + '_step:' + str(nStep) + '_rep:' + str(nRepeats)
        filenametxt = filename + '.txt'
        with open(filenametxt, 'w') as f:
            # Change the standard output to the file we created.
            sys.stdout = f
            print('parameters: clfD:', self.clfD,
                  'clfA: ', self.clfA, 
                  'Step: ', nStep, 
                  'repeats:', nRepeats)
            print("mean_tpr_aucDs = ", mean_tpr_aucDs)
            print("mean_utpr_aucDs = ", mean_utpr_aucDs)
            print("mean_afr_aucDs = ", mean_afr_aucDs)
            print("std_tpr_aucDs = ", std_tpr_aucDs)
            print("std_utpr_aucDs = ", std_utpr_aucDs)
            print("std_afr_aucDs = ", std_afr_aucDs)
        sys.stdout = original_stdout

    def Run_RTrainSize(self, listData, listLabel, nRepeats = 2, nSteps = 5, bBeth = False):
        self.mlistData = listData
        self.mlistLabel = listLabel
    
        np.random.seed(None)
        self.clfD = self.chooseClassifier(self.m_nclfD)
        clfD_name = ' D: ' + str(type(self.clfD)).split(".")[-1][:-2]

        self.clfA = self.chooseClassifier(self.m_nclfA)
        clfA_name = ' A: ' + str(type(self.clfA)).split(".")[-1][:-2]

        # repeat for different randomizations (trainSize)
        # initialize lists of mean and std_dev values, 
        # will hold one value for each trainSize
        mean_tpr_aucDs = []
        std_tpr_aucDs = []
        #mean_utpr_aucDs = []
        #std_utpr_aucDs = []
        mean_afr_aucDs = []
        std_afr_aucDs = []
 
        # arrTrainPercent = [100,100-step,100-2*step,...,2]/100
        arrTrainPercent = np.flip(np.arange(100, 0, -nSteps)/100)
        print(arrTrainPercent)

        # Here we are selecting training data with diffirent
        # training percentage
        for self.mTrainSize in arrTrainPercent:
            print(self.mTrainSize)
            tpr_aucDs = []
            #utpr_aucDs = []
            afr_aucDs = []  # list of results
            for i in range(nRepeats):  # add roc values to corresponding lists at each iteration
                roc_aucD, roc_aucA, afr_aucD = self.computeAUCsDA(bBeth)
                tpr_aucDs += [roc_aucD]
                #utpr_aucDs += [utpr_aucD]
                afr_aucDs += [afr_aucD]

            print("trainSize = ", self.mTrainSize, " repeats = ", nRepeats)
            print("np.mean(tpr_aucDs)", round(np.mean(tpr_aucDs), 3))
            #print("np.mean(utpr_aucDs)", round(np.mean(utpr_aucDs), 3))
            print("np.mean(afr_aucDs)", round(np.mean(afr_aucDs), 3))

            # compute means and stds of lists obtained in the previous for cycle
            # and accumulate results in the corresponding lists of means and st_devs
            mean_afr_aucDs += [np.mean(afr_aucDs)]
            #mean_utpr_aucDs += [np.mean(utpr_aucDs)]
            mean_tpr_aucDs += [np.mean(tpr_aucDs)]
            std_afr_aucDs += [np.std(afr_aucDs)]
            #std_utpr_aucDs += [np.std(utpr_aucDs)]
            std_tpr_aucDs += [np.std(tpr_aucDs)]
        # END FOR

        # SaveData
        arrColumns = ['train_percent', 
                      'mean_afr_aucDs', 'mean_tpr_aucDs', 
                      'std_afr_aucDs', 'std_tpr_aucDs']
    
        df = pd.DataFrame(columns=arrColumns)
        df['train_percent'] = np.array(arrTrainPercent).tolist()
        df['mean_afr_aucDs'] = np.array(mean_afr_aucDs).tolist()
        #df['mean_utpr_aucDs'] = np.array(mean_utpr_aucDs).tolist() # Not in use
        df['mean_tpr_aucDs'] = np.array(mean_tpr_aucDs).tolist()
        df['std_afr_aucDs'] = np.array(std_afr_aucDs).tolist()
        #df['std_utpr_aucDs'] = np.array(std_utpr_aucDs).tolist()
        df['std_tpr_aucDs'] = np.array(std_tpr_aucDs).tolist()

        # filename = '.\local-data\D-' + str(self.clfD).replace('()', '') + \
        #             '_A-' + str(self.clfA).replace('()', '') + \
        #             '_Steps-' + str(nSteps) + '_Rep-' + str(nRepeats)
        filename = self.m_strDirPath + \
                    '/D-' + str(self.m_nclfD) + '_A-' + str(self.m_nclfA) + \
                    '_Steps-' + str(nSteps) + '_Rep-' + str(nRepeats)
        df.to_csv(filename + '.csv', index=False)

        # plot means for tpr, utpr and afr (D wrt A)
        plt.figure(1)  # plot mean values for each randomation trainSize
        fig, ax = plt.subplots()
        
        #plt.plot(np.flip(arrTrainPercent), mean_afr_aucDs, label='afr_AUC for clfD')
        #plt.plot(arrTrainPercent, mean_afr_aucDs, label='afr_AUC for clfD')
        # plt.plot(np.flip(arrTrainPercent), mean_utpr_aucDs,
        #          label='utpr_AUC for clfD')
        
        ax.plot(arrTrainPercent, mean_afr_aucDs, label='afr_AUC for clfD')
        ax.plot(arrTrainPercent, mean_tpr_aucDs, label = 'tpr_AUC for clfD')
        ax.set_xticks(arrTrainPercent) 
        plt.xticks(rotation ='vertical')
        plt.legend()
        plt.xlabel('Training Percentage')
        plt.title(clfA_name + " - " + clfD_name)
        plt.grid(True)
        plt.savefig(filename + '_means.png')
        #plt.show()

        # plot std_devs for utpr and afr (D wrt A)
        plt.figure(2)  # plot std_dev values for each randomation trainSize
        fig, ax = plt.subplots()

        # plt.errorbar(np.flip(arrTrainPercent),
        #              mean_utpr_aucDs, std_utpr_aucDs, marker='^', label='utpr_AUC for clfD')
        #plt.errorbar(np.flip(arrTrainPercent), mean_afr_aucDs, std_afr_aucDs, marker='^', label='afr_AUC for clfD')
        ax.errorbar(arrTrainPercent, mean_afr_aucDs, std_afr_aucDs, marker='^', label='afr_AUC for clfD')
        ax.set_xticks(arrTrainPercent) 
        plt.xticks(rotation ='vertical')
        plt.legend()
        plt.xlabel('Training Percentage')

        plt.title(clfA_name + " - " + clfD_name)
        plt.grid(True)
        plt.savefig(filename + '_stds.png')
        #plt.show()

        plt.clf()

##############################################################################
#
def ROC_example(listData, listLabel):
    nclfA = 5  # AdaBoostClassifier
    nclfD = 2  # RandomForestClassifier
    nclfR = 4  # MLPClassifier

    objM = CModels(0.7, nclfA, nclfD, nclfR)
    objM.RunROC(listData, listLabel)

    strFileName = os.path.join(os.getcwd(), 'local-data\kyoto_roc.pdf')
    objM.PlotROC(strFileName)

def aROC_example(listData, listLabel):
    # relevant parameteres
    nclfD = 2
    nclfA = 2  # choose classifiers (Defender and Adversary)
    # fracValue = 0.8  # part of the training set to be actually used

    objM = CModels(0.7, nclfA, nclfD, -1)
    objM.RunAROC(listData, listLabel)
    strFileName = os.path.join(os.getcwd(), 'local-data\kyoto_aroc.pdf')
    objM.PlotAROC(strFileName)

def UTPR_AFR_example(listData, listLabel):
    objM = CModels(0.7, -1, -1, -1)
    objM.Run_UTPR_AFR(listData, listLabel)

def RV_example(listData, listLabel):
    nclfA = 5
    objM = CModels(0.7, nclfA, -1, -1)
    objM.Run_RV(listData, listLabel)

def RTrainSize_example(listData, listLabel):
    nclfD = 3
    nclfA = 5
    objM = CModels(0.7, nclfA, nclfD, -1)
    objM.Run_RTrainSize(listData, listLabel)

################################################################
if __name__ == '__main__':
    print(os.getcwd())
    strRoot = '/home/sandeep.gupta/2023 Project/wsKyoto/kyoto2015-12'

    try:
        # Get the dataset (Digit, Kyoto, Beth)
        cDATA = 'Kyoto'
        objDS = CDataset()

        if cDATA == 'Digit':
            listData, listLabel = objDS.PrepareMNISTDS()
            print(listData.shape)
            print(len(listLabel))
            # print(listLabels)
        elif cDATA == 'Kyoto':
            strFileName = '20151201.txt'
            strPath = os.path.join(strRoot, strFileName)
            listData, listLabel = objDS.GetKyotoDataset(strPath)

        ROC_example(listData, listLabel)
        aROC_example(listData, listLabel)
        UTPR_AFR_example(listData, listLabel)
        RV_example(listData, listLabel)
        RTrainSize_example(listData, listLabel)
    except Exception as e:
        print(e)
        # print(f"Unexpected {e=}, {type(e)=}")
        # print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno),
        #      type(e).__name__, e)
