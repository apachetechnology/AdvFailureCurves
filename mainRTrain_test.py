import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from models import CModels
from dataset import CDataset
from plots import CPlots

def Run_RTrain(strDirPath, fTestSize, listSelectedClassifier,
               listData, listLabel, nRepeats, nSteps):
    # Run Rtrain using below classifier combination
    # Beth RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5) #3
    # MLPClassifier(alpha=1, max_iter=100),  # 4
    # AdaBoostClassifier(),  # 5
    
    # 	3	4	5
    #3	33	34	35
    #4	43	44	45
    #5	53	54	55

    for nclfD in listSelectedClassifier:
        for nclfA in listSelectedClassifier:
            print('Combination - D: ', nclfD, 'A: ', nclfA)
            objM = CModels(strDirPath, fTestSize, nclfA, nclfD, -1)
            objM.Run_RTrainSize(listData, listLabel, nRepeats, nSteps)
            #break
        #break

def Run_RV(strDirPath, fTestSize, listSelectedClassifier,
           listData, listLabel, nRepeats, nSteps):
    for nclfA in listSelectedClassifier:
        objM = CModels(strDirPath, fTestSize, nclfA, -1, -1)
        objM.Run_RV(listData, listLabel, nRepeats, nSteps)
        #break


def Plot_RTrain_Results(strDirPath, listSelectedClassifier,
                        nSteps, nRepeats):
    # This is for testing purpose
    # strOutDir = '2023-08-15_10_52_13'
    # strDirPath = os.path.join(os.getcwd(), 'local-data', strOutDir)

    if len(listSelectedClassifier) < 3:
        print('Please select 3 classifiers.')
        return

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

    #
    df_11 = dictDF[(listSelectedClassifier[0], listSelectedClassifier[0])]
    arrTrainPercent = df_11.loc[:,['train_percent']].to_numpy()
    # print(df.loc[:,['train_percent']].to_numpy())
    print(arrTrainPercent)
    mean_afr_aucDs11 = df_11.loc[:,['mean_afr_aucDs']].to_numpy()

    mean_afr_aucDs_all = np.empty([mean_afr_aucDs11.shape[0], 0])
    for nclfD in listSelectedClassifier:
        for nclfA in listSelectedClassifier:
            df = dictDF[(nclfD, nclfA)]
            #print(df)
            mean_afr_aucDs_all = np.hstack((mean_afr_aucDs_all, df.loc[:,['mean_afr_aucDs']].to_numpy()))
            #break
        #break

    # print(mean_afr_aucDs_all)
    # print(mean_afr_aucDs_all[:,0:3])
    # print(mean_afr_aucDs_all[:,3:6])
    # print(mean_afr_aucDs_all[:,6:9])

    mean_afr_aucDss = np.mean(mean_afr_aucDs_all, axis=1)
    mean_afr_aucDs1 = np.mean(mean_afr_aucDs_all[:,0:3], axis=1)
    mean_afr_aucDs2 = np.mean(mean_afr_aucDs_all[:,3:6], axis=1)
    mean_afr_aucDs3 = np.mean(mean_afr_aucDs_all[:,6:9], axis=1)

    # mean_afr_aucRs is mean of mean_afr_aucDs for each classifier
    # obtained using post-learning randomization data from RV.py
    dictDF_RV = {}
    for nclfA in listSelectedClassifier:
        filename = strDirPath + \
                   '/A-' + str(nclfA) + \
                   '_Steps-' + str(nSteps) + '_Rep-' + str(nRepeats)
        print(filename + '.csv')
        dictDF_RV[nclfA] = pd.read_csv(filename + '.csv', delimiter=',')

    mean_afr_aucRs_all = np.empty([mean_afr_aucDs11.shape[0], 0])
    for nclfA in listSelectedClassifier:
        df_RV = dictDF_RV[nclfA]
        mean_afr_aucRs_all = np.hstack((mean_afr_aucRs_all, df_RV.loc[1:mean_afr_aucDs11.shape[0],['mean_afr_aucRs']].to_numpy()))

    #print(mean_afr_aucRs_all)
    mean_afr_aucRs = np.mean(mean_afr_aucRs_all, axis=1)

    fig, ax = plt.subplots()
    ax.plot(arrTrainPercent, mean_afr_aucDss, label='afr_AUC for all')
    ax.plot(arrTrainPercent, mean_afr_aucDs1, label='afr_AUC row1')
    ax.plot(arrTrainPercent, mean_afr_aucDs2, label='afr_AUC row2')
    ax.plot(arrTrainPercent, mean_afr_aucDs3, label='afr_AUC row3')
    ax.plot(arrTrainPercent, mean_afr_aucDs11, label='afr_AUC RF')
    plt.plot(arrTrainPercent, mean_afr_aucRs, label='afr_AUC for post-learn randomization')
    ax.set_xticks(arrTrainPercent) 
    ax.legend(loc='best')

    plt.xticks(rotation ='vertical')
    plt.grid(True)
    plt.savefig(strDirPath + '/Result.pdf', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.clf()

    #print results of 4 different anti-evasion algorithms
    maxAFRmatrix = max(mean_afr_aucDss)
    maxAFRrowByRow = (max(mean_afr_aucDs1) + max(mean_afr_aucDs2) + max(mean_afr_aucDs3))/3
    AFRrandomForest = mean_afr_aucDs11[mean_afr_aucDs11.shape[0]-1, 0]
    maxAFRpostLearn = max(mean_afr_aucRs)

    print("Max afr for mean of matrix: ", np.round(maxAFRmatrix, 2))
    print("Mean of max afr for all rows: ", np.round(maxAFRrowByRow,2))
    print("Simple use of Random forest for full trainset: ", np.round(AFRrandomForest,2))
    print("Max afr for post learning randomization: ", np.round(maxAFRpostLearn, 2))

    with open(strDirPath + '/Result.txt', 'w') as fp:
        fp.writelines('Classifiers: ' + str(listSelectedClassifier) + '\n')
        fp.writelines('Repeats: ' + str(nRepeats) + '\n')
        fp.writelines('Steps: ' + str(nSteps) + '\n')
        fp.writelines("Max afr for mean of matrix: " + str(np.round(maxAFRmatrix, 2)) + '\n')
        fp.writelines("Mean of max afr for all rows: " + str(np.round(maxAFRrowByRow,2)) + '\n')
        fp.writelines("Simple use of Random forest for full trainset: " + str(np.round(AFRrandomForest,2)) + '\n')
        fp.writelines("Max afr for post learning randomization: " + str(np.round(maxAFRpostLearn, 2)) + '\n')

################################################################
if __name__ == '__main__':
    print(os.getcwd())
    #strRoot = '/home/sandeep.gupta/2023 Project/wsKyoto/kyoto2015-12'

    # Get the dataset (Digit, Kyoto, Beth)
    listDataset = ['Digit', 'Kyoto', 'Beth']

    cDATA = 'Digit'
    objDS = CDataset()

    if cDATA == 'Digit':
        listData, listLabel = objDS.PrepareMNISTDS()
        print(listData.shape)
        print(len(listLabel))
        # print(listLabels)
        fTestSize = 0.7
        nRepeats = 1
    elif cDATA == 'Kyoto':
        strFileName =  r'./local-data/Kyoto2015DS.csv'
        #strPath = os.path.join(strRoot, strFileName)
        listData, listLabel = objDS.GetKyotoDataset(strFileName)
        fTestSize = 0.9
        nRepeats = 2
    elif cDATA == 'Beth':
        strFileName =  r'./local-data/BethDataset16Aug2023.csv'
        listData, listLabel = objDS.GetBethDataset(strFileName)
        fTestSize = 0.1656035
        nRepeats = 10

    nSteps = 5

    # Generate output directory
    strOutDir = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    print(strOutDir)
    strDirPath = os.path.join(os.getcwd(), 'local-data', cDATA + '_' + strOutDir)
    #os.makedirs(strDirPath)

    listSelectedClassifier = [3, 4, 5]
    # Run_RTrain(strDirPath, fTestSize, listSelectedClassifier,
    #            listData, listLabel, nRepeats, nSteps)
    # Run_RV(strDirPath, fTestSize, listSelectedClassifier,
    #        listData, listLabel, nRepeats, nSteps)
    # Plot_RTrain_Results(strDirPath, listSelectedClassifier, 
    #                     nSteps, nRepeats)

    print('DONE')