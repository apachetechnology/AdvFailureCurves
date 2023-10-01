import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class CAnalyzeResults:
    def __init__(self):
        print('CAnalyzeResults Object Created')

    def AnalyzeRTrain(self):
        dictDF = {}
        for nclfD in listSelectedClassifier:
            for nclfA in listSelectedClassifier:
                filename = strDirPath + \
                        '/D-' + str(nclfD) + '_A-' + str(nclfA) + \
                        '_Steps-' + str(nSteps) + '_Rep-' + str(nRepeats)
                print(filename + '.csv')
                dictDF[(nclfD, nclfA)] = pd.read_csv(filename + '.csv', delimiter=',')
            #END FOR
        #END FOR

        # Get the RF data from dictionary
        df_11 = dictDF[(listSelectedClassifier[0], listSelectedClassifier[0])]
        self.arrTrainPercent = df_11.loc[:,['train_percent']].to_numpy()
        # print(df.loc[:,['train_percent']].to_numpy())
        #print(arrTrainPercent)
        self.mean_afr_aucDs11 = df_11.loc[:,['mean_afr_aucDs']].to_numpy()
        self.mean_tpr_aucDs11 = df_11.loc[:,['mean_tpr_aucDs']].to_numpy()

        mean_afr_aucDs_all = np.empty([self.mean_afr_aucDs11.shape[0], 0])
        mean_tpr_aucDs_all = np.empty([self.mean_afr_aucDs11.shape[0], 0])
        for nclfD in listSelectedClassifier:
            for nclfA in listSelectedClassifier:
                df = dictDF[(nclfD, nclfA)]
                #print(df)
                mean_afr_aucDs_all = np.hstack((mean_afr_aucDs_all, df.loc[:,['mean_afr_aucDs']].to_numpy()))
                mean_tpr_aucDs_all = np.hstack((mean_tpr_aucDs_all, df.loc[:,['mean_tpr_aucDs']].to_numpy()))
                #break
            #break

        # print(mean_afr_aucDs_all)
        # print(mean_afr_aucDs_all[:,0:3])
        # print(mean_afr_aucDs_all[:,3:6])
        # print(mean_afr_aucDs_all[:,6:9])

        self.mean_afr_aucDss = np.mean(mean_afr_aucDs_all, axis=1)
        self.mean_tpr_aucDss = np.mean(mean_tpr_aucDs_all, axis=1)
        
        self.mean_afr_aucDs1 = np.mean(mean_afr_aucDs_all[:,0:3], axis=1)
        self.mean_afr_aucDs2 = np.mean(mean_afr_aucDs_all[:,3:6], axis=1)
        self.mean_afr_aucDs3 = np.mean(mean_afr_aucDs_all[:,6:9], axis=1)

        self.mean_tpr_aucDs1 = np.mean(mean_tpr_aucDs_all[:,0:3], axis=1)
        self.mean_tpr_aucDs2 = np.mean(mean_tpr_aucDs_all[:,3:6], axis=1)
        self.mean_tpr_aucDs3 = np.mean(mean_tpr_aucDs_all[:,6:9], axis=1)


    def GetMeanAFR_aucRs(self):
        # mean_afr_aucRs is mean of mean_afr_aucDs for each classifier
        # obtained using post-learning randomization data from RV.py
        dictDF_RV = {}
        for nclfA in listSelectedClassifier:
            filename = strDirPath + \
                    '/A-' + str(nclfA) + \
                    '_Steps-' + str(nSteps) + '_Rep-' + str(nRepeats)
            print(filename + '.csv')
            dictDF_RV[nclfA] = pd.read_csv(filename + '.csv', delimiter=',')

        df_1 = dictDF_RV[listSelectedClassifier[0]]
        self.arrRandomPercent = df_1.loc[:,['train_percent']].to_numpy() * 100
        nRows = self.arrRandomPercent.shape[0]

        mean_afr_aucRs_all = np.empty([nRows, 0])
        mean_tpr_aucRs_all = np.empty([nRows, 0])
        for nclfA in listSelectedClassifier:
            df_RV = dictDF_RV[nclfA]
            mean_afr_aucRs_all = np.hstack((mean_afr_aucRs_all, df_RV.loc[:,['mean_afr_aucRs']].to_numpy()))
            mean_tpr_aucRs_all = np.hstack((mean_tpr_aucRs_all, df_RV.loc[:,['mean_tpr_aucRs']].to_numpy()))

        #print(mean_afr_aucRs_all)
        self.mean_afr_aucRs = np.mean(mean_afr_aucRs_all, axis=1)
        self.mean_tpr_aucRs = np.mean(mean_tpr_aucRs_all, axis=1)


    def PrintResults(self):
        #print results of 4 different anti-evasion algorithms
        # Combined
        maxAFRrowByRow = (max(self.mean_afr_aucDs1) + max(self.mean_afr_aucDs2) + 
                          max(self.mean_afr_aucDs3))/3
        
        indexAFR_Row1 = np.array(self.mean_afr_aucDs1).argmax()
        indexAFR_Row2 = np.array(self.mean_afr_aucDs2).argmax()
        indexAFR_Row3 = np.array(self.mean_afr_aucDs3).argmax()

        
        maxTPRrowByRow = (self.mean_tpr_aucDs1[indexAFR_Row1] + 
                          self.mean_tpr_aucDs2[indexAFR_Row2] + 
                          self.mean_tpr_aucDs3[indexAFR_Row3])/3
        
        # Matrix 
        maxAFRmatrix = max(self.mean_afr_aucDss)
        indexAFRmatrix = np.array(self.mean_afr_aucDss).argmax()
        maxTPRmatrix = max(self.mean_tpr_aucDss)

        # Random Forest
        AFR_RF = self.mean_afr_aucDs11[self.mean_afr_aucDs11.shape[0]-1, 0]
        TPR_RF = self.mean_tpr_aucDs11[self.mean_tpr_aucDs11.shape[0]-1, 0]

        # PIN
        AFR_pinRF = max(self.mean_afr_aucDs11)
        indexAFR_Pin = self.mean_afr_aucDs11.argmax()
        #print('******************', self.mean_afr_aucDs11)

        TPR_pinRF = self.mean_tpr_aucDs11[indexAFR_Pin]

        # TSET
        AFRtsRF = np.mean(self.mean_afr_aucDs11)
        TPRtsRF = np.mean(self.mean_tpr_aucDs11)
        
        # Post Learn
        maxAFRpostLearn = max(self.mean_afr_aucRs)
        indexAFRpostLearn = np.array(self.mean_afr_aucRs).argmax()
        maxTPRpostLearn = max(self.mean_tpr_aucRs)

        print('\nAFR-AUC\n')
        print("Combined: Mean of max afr for all rows: ", np.round(maxAFRrowByRow, 2))
        print("Matrix: Max afr for mean of matrix: ", np.round(maxAFRmatrix, 2))
        print("Pin: Simple use of Random forest, pinning (max): ", np.round(AFR_pinRF,2))
        print("RF: Simple use of Random forest for full trainset: ", np.round(AFR_RF,2))
        print("TSet: Simple use of Random forest, avg: ", np.round(AFRtsRF,2))
        print("Post-learn: Max afr for post learning randomization: ", np.round(maxAFRpostLearn, 2))

        print('\nTPR\n')
        print("Combined: Mean of max afr for all rows: ", np.round(maxTPRrowByRow, 2))
        print("Matrix: Max ROC-AUC for mean of matrix: ", np.round(maxTPRmatrix, 2))
        print("Pin: Simple use of Random forest, pinning (max): ", np.round(TPR_pinRF,2), ' index ' , indexAFR_Pin)
        print("RF: Simple use of Random forest for full trainset: ", np.round(TPR_RF,2))
        print("TSet: Simple use of Random forest, avg: ", np.round(TPRtsRF,2))
        print("Post-learn: Max ROC-AUC for post learning randomization: ", np.round(maxTPRpostLearn, 2))

        with open(strDirPath + '/Result' + strToken + '.txt' , 'w') as fp:
            fp.writelines('Classifiers: ' + str(listSelectedClassifier) + '\n')
            fp.writelines('Repeats: ' + str(nRepeats) + '\n')
            fp.writelines('Steps: ' + str(nSteps) + '\n')
            
            fp.writelines('\nAFR-AUC**************\n')

            fp.writelines("Combined: Mean of max afr for all rows: " +  str(np.round(100*maxAFRrowByRow,2)) + '\n')
            fp.writelines("Matrix: Max afr for mean of matrix: " + str(np.round(100*maxAFRmatrix, 2)) + '\n')
            fp.writelines("Pin: Simple use of Random forest, pinning (max): " + str(np.round(100*AFR_pinRF,2)) + '\n')
            fp.writelines("RF: Simple use of Random forest for full trainset: " + str(np.round(100*AFR_RF,2)) + '\n')
            fp.writelines("TSet: Simple use of Random forest, avg: " + str(np.round(100*AFRtsRF,2)) + '\n')
            fp.writelines("Post-learn: Max afr for post learning randomization: " + str(np.round(100*maxAFRpostLearn, 2)) + '\n')

            fp.writelines('\nTPR**************\n')
            #fp.writelines("Combined: Mean of max afr for all rows: " +  str(np.round(100*maxTPRrowByRow,2)) + '\n')
            #fp.writelines("Matrix: Max ROC-AUC for mean of matrix: " + str(np.round(100*maxTPRmatrix, 2)) + '\n')
            fp.writelines("Pin: Simple use of Random forest, pinning (max): " + str(np.round(100*TPR_pinRF,2)) + '\n')
            fp.writelines("RF: Simple use of Random forest for full trainset: " + str(np.round(100*TPR_RF,2)) + '\n')
            fp.writelines("TSet: Simple use of Random forest, avg: " + str(np.round(100*TPRtsRF,2)) + '\n')
            #fp.writelines("Post-learn: Max ROC-AUC for post learning randomization: " + str(100*np.round(maxTPRpostLearn, 2)) + '\n')

            fp.writelines('\nTPR-ROC values at corresponding index for maximum AFR-AUC\n')
            fp.writelines('Combined\tMatrix\tpin\tRF\tTset\tPost-learn\n')
            fp.writelines(
                          str(np.round(100*maxTPRrowByRow, 2)) + '\t' +
                          str(np.round(100*self.mean_tpr_aucDss[indexAFRmatrix], 2)) + '\t' +
                          str(np.round(100*TPR_pinRF,2)) + '\t' +
                          str(np.round(100*TPR_RF,2)) + '\t' +
                          str(np.round(100*TPRtsRF,2)) + '\t' +
                          str(np.round(100*self.mean_tpr_aucRs[indexAFRpostLearn], 2))
                          )

    def PlotResults(self, strTitle):
        fig, axs = plt.subplots(2, figsize=(6, 7)) # figsize=(6, 3), dpi=300
        #fig.suptitle(strTitle)
        plt.subplots_adjust(hspace=.55)

        strL01 = 'AFR-AUC for post-learn randomization'
        l01, = axs[0].plot(self.arrRandomPercent, self.mean_afr_aucRs, 
                      label=strL01)
        axs[0].set_xticks(self.arrRandomPercent.ravel())
        plt.setp(axs[0].get_xticklabels(), rotation=90, ha='center')
        axs[0].set_xlabel('Random data percentage')
        axs[0].set_ylabel('AFR-AUC')
        #ax1.set_yticks(np.arange(0, 1.1, 0.1))
        axs[0].grid(linestyle='dotted')

        ax02 = axs[0].twiny()
        ax02.invert_xaxis()
        # Process the arrays
        #arrTrainPercentNew = np.append(0, self.arrTrainPercent.ravel())
        #mean_afr_aucDs11New = np.append(None, self.mean_afr_aucDs11.ravel())
        #mean_afr_aucDssNew = np.append(None, self.mean_afr_aucDss.ravel())

        strL02 = 'AFR-AUC for Random Forest'
        l02, = ax02.plot(self.arrTrainPercent, self.mean_afr_aucDs11, 
                 label=strL02, color='orange')
        strL03 = 'average AFR-AUC for model matrix'
        l03, = ax02.plot(self.arrTrainPercent, self.mean_afr_aucDss, 
                 label=strL03, color='green')
        
        #ax2.plot(arrTrainPercent, mean_afr_aucDs1, label='AFR-AUC row1')
        #ax2.plot(arrTrainPercent, mean_afr_aucDs2, label='AFR-AUC row2')
        #ax2.plot(arrTrainPercent, mean_afr_aucDs3, label='AFR-AUC row3')
        
        ax02.set_xticks(self.arrTrainPercent)
        plt.setp(ax02.get_xticklabels(), rotation=90, ha='center')
        ax02.set_xlabel('Training data size')
        plt.figlegend([l01, l02, l03], [strL01, strL02, strL03], 
                      loc=(.4, .6), fontsize=8)
        
        # TPR plots
        strL11 = 'ROC-AUC for post-learn randomization'
        l11, = axs[1].plot(self.arrRandomPercent, self.mean_tpr_aucRs, 
                      label=strL11)
        axs[1].set_xticks(self.arrRandomPercent.ravel())
        plt.setp(axs[1].get_xticklabels(), rotation=90, ha='center')
        axs[1].set_xlabel('Random data percentage')
        axs[1].set_ylabel('ROC-AUC')
        axs[1].grid(linestyle='dotted')

        ax12 = axs[1].twiny()
        ax12.invert_xaxis()

        strL12 = 'ROC-AUC for Random Forest'
        l12, = ax12.plot(self.arrTrainPercent, self.mean_tpr_aucDs11, 
                 label=strL12, color='orange')
        strL13 = 'average ROC-AUC for model matrix'
        l13, = ax12.plot(self.arrTrainPercent, self.mean_tpr_aucDss, 
                label=strL13, color='green')
        ax12.set_xticks(self.arrTrainPercent)
        plt.setp(ax12.get_xticklabels(), rotation=90, ha='center')
        ax12.set_xlabel('Training data size')
        plt.figlegend([l11, l12, l13], [strL11, strL12, strL13], 
                      loc=(.3, .2), fontsize=8)
        
        #plt.figlegend(loc = 'lower center', ncol=2, labelspacing=0.)

        #plt.title(strTitle, y=2.5)
        #fig.legend(loc='lower center') # X, Ys

        plt.savefig(strDirPath + '/Result' + strToken + '.pdf', dpi=300, bbox_inches='tight')
        #plt.show()
        plt.clf()

    def PlotResults_Two(self, strTitle):
        fig, axs = plt.subplots(2)
        fig.suptitle(strTitle)
        plt.subplots_adjust(hspace=0.4)

        axs[0].plot(self.arrRandomPercent, self.mean_afr_aucRs, label='AFR-AUC for post-learn randomization')
        axs[0].set_xticks(self.arrRandomPercent.ravel())
        plt.setp(axs[0].get_xticklabels(), rotation=90, ha='center')
        axs[0].set_xlabel('Random data percentage')
        #axs[0].set_yticks(np.arange(0, 1.1, 0.1))
        axs[0].legend(loc='lower right')
        axs[0].grid(True)

        axs[1].invert_xaxis()

        axs[1].plot(self.arrTrainPercent, self.mean_afr_aucDs11, 
                 label='afr-AUC for Random Forest', color='orange')
        axs[1].plot(self.arrTrainPercent, self.mean_afr_aucDss, 
                 label='average afr-AUC for model matrix', color='green')
        #axs[1].plot(arrTrainPercent, mean_afr_aucDs1, label='AFR-AUC row1')
        #axs[1].plot(arrTrainPercent, mean_afr_aucDs2, label='AFR-AUC row2')
        #axs[1].plot(arrTrainPercent, mean_afr_aucDs3, label='AFR-AUC row3')
        
        axs[1].set_xticks(self.arrTrainPercent)
        plt.setp(axs[1].get_xticklabels(), rotation=90, ha='center')
        axs[1].set_xlabel('Training data size')
        axs[1].legend(loc='lower right')
        #axs[1].set_yticks(np.arange(0, 1.1, 0.1))
        axs[1].grid(True)

        plt.savefig(strDirPath + '/Result.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()

    def Plot_RTrain_Results(self, strTitle):
        if len(listSelectedClassifier) < 3:
            print('Please select 3 classifiers.')
            return

        self.GetMeanAFR_aucRs()
        self.AnalyzeRTrain()

        self.PrintResults()
        self.PlotResults(strTitle)
        
#---------------------------------------------------------------------------------
################################################################
if __name__ == '__main__':
    print(os.getcwd())

    # Specify parameters - Same as used to run the experiment
    listSelectedClassifier = [3, 4, 5]
    nSteps = 5

    obj = CAnalyzeResults()

    strToken = '_Digit'
    nRepeats = 100
    strDirPath = './Results25Aug/Digit_2023-08-21_12_32_49'
    obj.Plot_RTrain_Results('Digit, Total: (1797, 64) Training: (539, 64)')

    strToken = '_Kyoto'
    nRepeats = 10
    strDirPath = './Results25Aug/Kyoto_2023-08-21_14_36_44'
    obj.Plot_RTrain_Results('Kyoto, Total: (60000, 13) Training: (6000, 13)')

    strToken = '_Beth_OoS'
    nRepeats = 10
    strDirPath = './Results25Aug/Beth_2023-08-23_11_37_54_OoS'
    obj.Plot_RTrain_Results('Beth Total: (1026970, 6) Training: (856900, 6)')

    strToken = '_Beth_IS'
    nRepeats = 10
    strDirPath = './Results25Aug/Beth_2023-08-22_14_50_55_IS'
    obj.Plot_RTrain_Results('Beth Total: (1026970, 6) Training: (102697, 6)')