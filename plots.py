
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

######################################################################################
#
class CPlots:
    def __init__(self, testPos, testNeg):
        #print('CPlots Object Created')
        self.testPos = testPos
        self.testNeg = testNeg
        #print('CPlots::Init Positive:', self.testPos, 'Negative:', self.testNeg)

    def getThreshold(self, scores):
        thresholds = np.unique(scores)
        thresholds = thresholds[::-1]  # invert order, then add first element
        thresholds = np.concatenate(([thresholds[0]*2], thresholds))
        return (thresholds)

    # define myroc_curve
    # similar to metrics.roc_curve
    def myroc_curve(self, y, scores, thresholds):
        # false positive numbers for all thresholds
        fp = np.zeros(len(thresholds))
        # false negative numbers for all thresholds
        fn = np.zeros(len(thresholds))
        for j in np.arange(len(thresholds)):
            fp[j] = 0
            fn[j] = 0  # initialize fp and fn to 0 for threshold j
            fp[j] = np.logical_and(y == 0, scores >= thresholds[j]).sum()
            fn[j] = np.logical_and(y == 1, scores < thresholds[j]).sum()
        return (fp, fn)  # return lists of false positive and false negative

    # Slower not in used
    def myroc_curve_forloop(self, y, scores, thresholds):
        fp = np.arange(len(thresholds))
        fn = np.arange(len(thresholds))
        # more efficient version in Beth case study
        for j in np.arange(len(thresholds)):
            fp[j] = 0
            fn[j] = 0
            for i in np.arange(len(y)):
                if y[i] == 0 and scores[i] >= thresholds[j]:
                    fp[j] += 1
                else:
                    if y[i] == 1 and scores[i] < thresholds[j]:
                        fn[j] += 1
        return (fp, fn)

    def plot_tpr(self, pred, y_test, clf_name):
        t = self.getThreshold(pred)
        fp, fn = self.myroc_curve(y_test, pred, t)
        fpr = fp/self.testNeg
        tpr = (self.testPos-fn)/self.testPos
        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr,
                 label='TPR ' + clf_name + ' (AUC='+str(round(roc_auc, 2))+')')

    # adversarial roc curve
    def aroc_curve(self, y, scores, ascores):
        thresholds = self.getThreshold(scores)
        fp = np.zeros(len(thresholds))   # false defender positives
        afn = np.zeros(len(thresholds))  # adversarial false negatives
        kfn = np.zeros(len(thresholds))  # known false negatives

        for j in np.arange(len(thresholds)):
            fp[j] = 0
            afn[j] = 0  # false defender & adversarial negatives
            kfn[j] = 0  # known false negatives
            fp[j] = np.logical_and(y == 0, scores >= thresholds[j]).sum()
            cond2 = np.logical_and(y == 1, ascores < thresholds[j])
            afn[j] = cond2.sum()
            kfn[j] = np.logical_and(cond2, scores < thresholds[j]).sum()

        return (fp, kfn, afn)  # see slide 5 of 2.adversarialROCcurves.ppt

    # define how to compute and plot adversarial roc curves (utpr and afr)
    def plot_utpr(self, fp, kfn, clf_name):
        fpr = fp/self.testNeg
        kfnr = kfn/self.testPos
        utpr = 1-kfnr
        aroc_auc = round(metrics.auc(fpr, utpr), 2)
        plt.plot(fpr, utpr, label = 'utpr ' + clf_name + ' (AUC=' + str(aroc_auc) + ')')
        return (utpr)
    
    def plot_afr(self, fp, kfn, afn, clf_name):
        fpr = fp/self.testNeg

        with np.errstate(divide='ignore', invalid='ignore'):
            asr = np.nan_to_num(kfn/afn)

        afr = 1-asr
        aroc_auc = round(metrics.auc(fpr, afr), 2)
        plt.plot(fpr, afr, label = 'afr '+ clf_name +' (AUC=' + str(aroc_auc) + ')')
        return (afr)
    
    # compute fpr,tpr,roc_auc (later needed for plots)
    def compute_tprAUROC(self, pred, y_test):  # compute tpr & roc_AUC for pred
        t = self.getThreshold(pred)
        fp, fn = self.myroc_curve(y_test, pred, t)
        fpr = fp/self.testNeg
        tpr = (self.testPos - fn)/self.testPos
        roc_auc = metrics.auc(fpr, tpr)

        return (fpr, tpr, roc_auc)
    
    def compute_tprAUROC_RV(self, pred, y_test): 
        t = self.getThreshold(pred)
        fp, fn = self.myroc_curve(y_test, pred, t)
        fpr = fp/self.testNeg
        tpr = (self.testPos - fn)/self.testPos
        roc_auc = metrics.auc(fpr, tpr)

        return(roc_auc)

    # compute fpr, utpr & utpr_AUC for predD/predR wrt predA (given fp, kfn)
    def compute_utprAUROC(self, fp, kfn):
        fpr = fp/self.testNeg
        kfnr = kfn/self.testPos
        utpr = 1-kfnr
        aroc_auc = round(metrics.auc(fpr, utpr), 2)
        return (fpr, utpr, aroc_auc)
    
    def compute_utprAUROC_RV(self, fp, kfn):
        fpr = fp/self.testNeg
        kfnr = kfn/self.testPos
        utpr = 1-kfnr
        aroc_auc = round(metrics.auc(fpr, utpr), 2)
        return aroc_auc
    
    # compute fpr, afr & afr_AUC for predD/predR wrt predA (given fp, kfn, afn)
    def compute_afrAUROC(self, fp, kfn, afn):
        fpr = fp/self.testNeg

        with np.errstate(divide='ignore', invalid='ignore'):
            asr = np.nan_to_num(kfn/afn)

        afr = 1-asr
        aroc_auc = round(metrics.auc(fpr, afr), 2)
        return (fpr, afr, aroc_auc)
    
    def compute_afrAUROC_RV(self, fp, kfn, afn):
        fpr = fp/self.testNeg

        with np.errstate(divide='ignore', invalid='ignore'):
            asr = np.nan_to_num(kfn/afn)

        afr = 1-asr
        aroc_auc = round(metrics.auc(fpr, afr), 2)
        return aroc_auc
    
    def arocPlot(self, fpr, yaxis, aroc_auc, clf_name):  # plot utpr or afr curves
        plt.plot(fpr, yaxis, label=clf_name+' (AUC='+str(aroc_auc)+')')

    #define arocPlotCol, then plot mean tpr for clfD and clfR
    # same as arocPlot, with colours
    def arocPlotCol(self, fpr, utpr, aroc_auc, clf_name, col):
        plt.plot(fpr, utpr, label=clf_name + ' (AUC='+str(aroc_auc) + ')', color=col)