import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from Core.models import CModels
from Core.dataset import CDataset
from Core.plots import CPlots

# Run Rtrain using below classifier combination
    # Beth RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5) #3
    # MLPClassifier(alpha=1, max_iter=100),  # 4
    # AdaBoostClassifier(),  # 5
    
    # 	3	4	5
    #3	33	34	35
    #4	43	44	45
    #5	53	54	55
def Run_RTrain(strDirPath, fTestSize, listDefClassifier, listAdvClassifier,
               listData, listLabel, nEPOCHS, nSteps, bBethOOS,
               bShuffle=False, aRS=None):
    # Defender's vs Adversary's Capability
    for nclfD in listDefClassifier:
        for nclfA in listAdvClassifier:
            print('Combination - D: ', nclfD, 'A: ', nclfA)
            objM = CModels(strDirPath, fTestSize, 
                           nclfA, nclfD, -1,
                           bShuffle, aRS)
            objM.Run_RTrainSize(listData, listLabel, nEPOCHS, nSteps, bBethOOS)
            #break
        #break

def Run_RV(strDirPath, fTestSize, listAdvClassifier,
           listData, listLabel, nEPOCHS, nSteps, bBethOOS):
    for nclfA in listAdvClassifier:
        print('Adversary - A:', nclfA)
        dtStart = datetime.now()
        objM = CModels(strDirPath, fTestSize, nclfA, -1, -1,
                       bShuffle=False, aRS=None)
        objM.Run_RV(listData, listLabel, nEPOCHS, nSteps, bBethOOS)
        cpu_time = datetime.now() - dtStart
        print(f'Time elapsed:{cpu_time/60.0} minutes')
        #break

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
        nEPOCHS = 1
    elif cDATA == 'Kyoto':
        strFileName =  r'./local-data/Kyoto2015DS.csv'
        #strPath = os.path.join(strRoot, strFileName)
        listData, listLabel = objDS.GetKyotoDataset(strFileName)
        fTestSize = 0.9
        nEPOCHS = 10
    elif cDATA == 'Beth':
        strFileName =  r'../DATA/Beth_16Aug2023.csv'
        listData, listLabel = objDS.GetBethDataset(strFileName)
        fTestSize = 0.1656035
        nEPOCHS = 10

    nSteps = 5

    # Generate output directory
    strOutDir = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    print(strOutDir)
    strDirPath = os.path.join(os.getcwd(), 'local-data', cDATA + '_' + strOutDir)
    #os.makedirs(strDirPath)

    listDefClassifier = [3, 4, 5]
    listAdvClassifier = [3, 4, 5]
    # Run_RTrain(strDirPath, fTestSize, listSelectedClassifier,
    #            listData, listLabel, nEPOCHS, nSteps)
    # Run_RV(strDirPath, fTestSize, listSelectedClassifier,
    #        listData, listLabel, nEPOCHS, nSteps)
    # Plot_RTrain_Results(strDirPath, listSelectedClassifier, 
    #                     nSteps, nEPOCHS)

    print('DONE')