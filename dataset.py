import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import datasets

#############################################################
# Class CDataset
class CDataset:
    def __init__(self):
        print('CDataset Object Created')

    def WriteFile(self, strFileName, listData):
        df = pd.DataFrame(listData)
        df.to_csv(strFileName, header=False, index=False)

    # Get Digits dataset & modification
    def PrepareDigitDS(self, bVerbose = False):
        np.random.seed(None)

        dsData = datasets.load_digits()
        n_samples = len(dsData.images)
        listData = dsData.images.reshape((n_samples, -1))
        if bVerbose:
            strFileName = os.path.join(os.getcwd(), 'local-data\digits.csv')
            print(strFileName)
            self.WriteFile(strFileName, listData)

        anomaly = 3  # consider this as positive (anomaly) & other digits as negative
        # count how many anomalies in digits.target
        anomalies = 0
        for i in np.arange(dsData.target.size):
            if dsData.target[i] == anomaly:
                dsData.target[i] = 1
            else:
                dsData.target[i] = 0

        # so now we just have two classes: "1"=anomalous and "0"=normal

        return listData, dsData.target
    
    # Beth Dataset
    def PrepareBethDataset(self):
        # prepare beth training and test set
        dftrain1 = pd.read_csv(os.path.join(os.getcwd(), 'local-data', 'Beth', 'Beth_train4.csv'))
        dfvalid1 = pd.read_csv(os.path.join(os.getcwd(), 'local-data', 'Beth', 'Beth_valid4.csv'))
        dftest1 = pd.read_csv(os.path.join(os.getcwd(), 'local-data', 'Beth', 'Beth_test4.csv'))

        print('Original: ', dftrain1.shape, dfvalid1.shape, dftest1.shape)

        # reduce size to make it faster if needed
        dftrain2 = dftrain1.sample(frac=0.9)
        dfvalid2 = dfvalid1.sample(frac=0.9)

        # Combine train and validatie files
        frames = [dftrain2, dfvalid2]
        df_train = pd.concat(frames, ignore_index=True)
        print(df_train.groupby('sus').size())

        dfYTrain = np.ravel(df_train['sus']) # 
        print('Train label: ', dfYTrain.shape)

        dfXTrain = df_train.drop("evil", axis=1)
        #dfXTrain = dfXTrain.drop("sus", axis=1)
        dfXTrain.drop_duplicates()
        print('Training dataset: ', dfXTrain.shape)
        #dfXTrain.to_csv('Beth_train.csv', index=False)


        dfTest_Sampled = dftest1.sample(frac=0.9)
        print(dfTest_Sampled.groupby('sus').size())
        
        # other relevant parameteres
        # fracValue=1.0 #part of the training set to be actually used prepare test set
        # ytest = np.ravel(df['evil']);
        dfYTest = np.ravel(dfTest_Sampled['sus'])
        print('Test Label', dfYTest.shape)
        
        # drop both targets (sus) and (evil, not used here) from test
        dfXTest = dfTest_Sampled.drop("evil", axis=1)
        #dfXTest = dfXTest.drop("sus", axis=1)
        dfXTest.drop_duplicates()
        print('Test Data', dfXTest.shape)
        #dfXTest.to_csv('Beth_test.csv', index=False)

        frames1 = [dfXTrain, dfXTest]
        dfCombined = pd.concat(frames1, ignore_index=True)
        dfCombined.to_csv('BethDataset16Aug2023.csv', index=False)

    def GetBethDataset(self, strFileName):
        df = pd.read_csv(strFileName, delimiter=',')
        dfData = df.loc[:,['processId','parentProcessId', 'mountNamespace',
                            'eventId', 'argsNum', 'returnValue']]

        dfLabel = df.loc[:,['sus']]
        print(dfLabel.groupby('sus').size())

        listData = dfData.to_numpy()
        listLabel = dfLabel.to_numpy()
        return listData, listLabel

    # Get Kyoto Dataset
    def PrepareSelectedKyotoDS(self, strFileName, strOutName):
        #strFilePath = os.path.join(os.getcwd(), strFileName)

        df = pd.read_csv(strFileName, header=None, delimiter='\t')
        df.columns = ['duration', 'service', 'source_bytes', 'destination_bytes', 
                      'count', 'same_srv_rate', 'serror_rate', 'srv_serror_rate', 
                      'dst_host_count', 'dst_host_srv_count', 'dst_host_same_src_port_rate',
                      'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'flag', 
                      'ids_detection', 'malware_detection', 'ashula_detection', 'label', 
                      'source_ip_address', 'source_port_number', 'destination_ip_address', 
                      'destination_port_number', 'start_time', 'protocol']
        
        df.drop_duplicates()
        #print(df.head())

        df = df.replace(1, 0)
        df = df.replace(-1, 1)
        df = df.replace(-2, 1)

        dfData = df.loc[:,['duration','source_bytes', 'source_port_number',
                            'count', 'same_srv_rate', 'serror_rate', 'srv_serror_rate',
                            'dst_host_count', 'dst_host_srv_count', 
                            'dst_host_same_src_port_rate', 'dst_host_serror_rate',
                            'destination_bytes', 'destination_port_number', 'label']]
        
        #dfLabel = df.loc[:,['label']]
        # 1 means the session was normal, 
        # -1 means known attack was observed in the session and
        # -2 means unknown attack was observed in the session.

        # Replace 1 with 0 and replace -1 & -2 with 1
        # dfLabel = dfLabel.replace(1, 0)
        # dfLabel = dfLabel.replace(-1, 1)
        # dfLabel = dfLabel.replace(-2, 1)

        # frames = [dfData, dfLabel]
        # df = pd.concat(frames)
        print(dfData.groupby('label').size())

        dfData.to_csv('./local-data/' + strOutName + '.csv', index=False)

    def WriteSelectedKyotoDS(self):
        strFileName = r'../Kyoto2015-12/20151201.txt'
        #self.PrepareSelectedKyotoDS(strFileName, '20151201')

        strFileName = r'../Kyoto2015-12/20151215.txt'
        self.PrepareSelectedKyotoDS(strFileName, '20151215')

        strFileName = r'../Kyoto2015-12/20151231.txt'
        #self.PrepareSelectedKyotoDS(strFileName, '20151231')

    def CreateBalanceDS(self, strFileName):
        df_01 = pd.read_csv(r'./local-data/Kyoto2015-12/20151201.csv', delimiter=',')
        print(df_01.groupby('label').size())

        df_01_Normal = df_01.loc[df_01['label'] == 0]
        df_01_Attack = df_01.loc[df_01['label'] == 1]
        df_01_Attack = df_01_Attack.sample(n=10000)

        df_15 = pd.read_csv(r'./local-data/Kyoto2015-12/20151215.csv', delimiter=',')
        print(df_15.groupby('label').size())

        df_15_Normal = df_15.loc[df_15['label'] == 0]
        df_15_Attack = df_15.loc[df_15['label'] == 1]
        df_15_Attack = df_15_Attack.sample(n=10000)

        df_31 = pd.read_csv(r'./local-data/Kyoto2015-12/20151231.csv', delimiter=',')
        print(df_31.groupby('label').size())

        df_31_Normal = df_31.loc[df_31['label'] == 0]
        df_31_Normal = df_31_Normal.sample(n=18052)
        df_31_Attack = df_31.loc[df_31['label'] == 1]
        df_31_Attack = df_31_Attack.sample(n=10000)

        frames = [df_01_Normal, df_15_Normal, df_31_Normal,
                  df_01_Attack, df_15_Attack, df_31_Attack]
        df = pd.concat(frames)
        print(df.shape)
        df.to_csv(strFileName, index=False)

    def GetKyotoDataset(self, strFileName):
        df = pd.read_csv(strFileName, delimiter=',')
        dfData = df.loc[:,['duration','source_bytes', 'source_port_number',
                            'count', 'same_srv_rate', 'serror_rate', 'srv_serror_rate',
                            'dst_host_count', 'dst_host_srv_count', 
                            'dst_host_same_src_port_rate', 'dst_host_serror_rate',
                            'destination_bytes', 'destination_port_number']]

        dfLabel = df.loc[:,['label']]
        print(dfLabel.groupby('label').size())

        listData = dfData.to_numpy()
        listLabel = dfLabel.to_numpy()
        return listData, listLabel

################################################################
if __name__ == '__main__':

    objDS = CDataset()
    #objDS.WriteSelectedKyotoDS()

    # DIGIT
    listData, listLabel = objDS.PrepareDigitDS()
    print(len(listLabel))
    # print(listLabels)
    
    fTestSize = 0.7
    X_trainALL, X_test, y_trainALL, y_test = train_test_split(
            listData, listLabel.ravel(), 
            test_size=fTestSize,  random_state = 42)
    print('Total:' , listData.shape, 'Training:', X_trainALL.shape)

    unique, counts = np.unique(y_trainALL, return_counts=True)
    print(unique, counts)

    unique, counts = np.unique(y_test, return_counts=True)
    print(unique, counts)
    

    # KYOTO
    strFileName = r'./local-data/Kyoto2015DS.csv'
    #objDS.CreateBalanceDS(strFileName)

    listData, listLabel = objDS.GetKyotoDataset(strFileName)
    print('Kyoto: ', listData.shape, listLabel.shape)
    
    fTestSize = 0.9
    X_trainALL, X_test, y_trainALL, y_test = train_test_split(
            listData, listLabel.ravel(), 
            test_size=fTestSize,  random_state = 42)
    print('Total:' , listData.shape, 'Training:', X_trainALL.shape)

    unique, counts = np.unique(y_trainALL, return_counts=True)
    print(unique, counts)

    unique, counts = np.unique(y_test, return_counts=True)
    print(unique, counts)

    # BETH
    strFileName = r'./local-data/BethDataset16Aug2023.csv'
    listData, listLabel = objDS.GetBethDataset(strFileName)
    print(listData.shape, listLabel.shape)

    fTestSize = 0.9
    X_trainALL, X_test, y_trainALL, y_test = train_test_split(
            listData, listLabel.ravel(), 
            test_size=fTestSize,  random_state = 42)
    print('Total:' , listData.shape, 
          'Training:', X_trainALL.shape, 
          'Testing:', X_test.shape)

    print(X_trainALL.shape, X_test.shape)
    unique, counts = np.unique(y_trainALL, return_counts=True)
    print(unique, counts)

    unique, counts = np.unique(y_test, return_counts=True)
    print(unique, counts)

    fTestSize = 0.1656035
    X_trainALL, X_test, y_trainALL, y_test = train_test_split(
            listData, listLabel.ravel(), 
            #test_size=fTestSize,  random_state = 42)
            test_size=fTestSize, shuffle = False)
    print('Total:' , listData.shape, 
          'Training:', X_trainALL.shape,
          'Testing:', X_test.shape)
    
    print(X_trainALL.shape, X_test.shape)
    unique, counts = np.unique(y_trainALL, return_counts=True)
    print(unique, counts)

    unique, counts = np.unique(y_test, return_counts=True)
    print(unique, counts)


    
