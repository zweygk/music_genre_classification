import pandas as pd
import numpy as np

def Import_Data(data_file, label_file):
    data = pd.read_csv(data_file,header=None)
    labels = pd.read_csv(label_file,header=None)
    debug_msg = str('Imported data '+str(data.shape)+' and labels '+str(labels.shape)+'.')
    print(debug_msg)
    return data, labels

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import imblearn
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks, InstanceHardnessThreshold, OneSidedSelection
from sklearn.preprocessing import StandardScaler, scale

def Remove_Zero_Variance(data):
    selector = VarianceThreshold()
    clean_data = selector.fit_transform(data)
    debug_msg = str('Zero variance features removed from data. Input shape: '+ str(data.shape)+'. Output shape: '+str(clean_data.shape)+'.')
    print(debug_msg)
    return clean_data

def Select_Features_From_Model(data, labels, model_name='LassoCV'):
    if model_name == 'LassoCV':
        model = SelectFromModel(LassoCV())
        model.fit(data, labels)
        selected_features = model.transform(data)
        debug_msg = str('Selected best features from input. Input shape: '+str(data.shape)+'. Output shape: '+str(selected_features.shape)+'.')
        print(debug_msg)
        return selected_features
        
def Normalize(data, type_string):
    if type_string == 'min-max':
        z = (data - data.min()) / (data.max() - data.min())
    elif type_string == 'z-score':
        z = (data-data.mean())/data.std()
    elif type_string == 'tanh':
        z = 0.5*(np.tanh(0.01*(data-data.mean())/data.std())+1)
    else:
        z = (data - data.mean()) / (data.max() - data.min())
    debug_msg = str('Data normalized using ' + type_string + ' method. Range: ['+ str(z.min())+', '+str(z.max())+'].')
    print(debug_msg)
    return z

def Standardize(data):
    sc = StandardScaler()
    X_std = sc.fit_transform(data)
    return X_std

from sklearn.model_selection import train_test_split

def Split_Data(data, labels, ratio=0.3, state=213):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=ratio, random_state=state)
    debug_msg = 'Data successfully split. Test data ratio = '+str(ratio)
    print(debug_msg)
    return X_train, X_test, y_train, y_test

def Shuffle(data,labels):
    shuffled_data, shuffled_labels = shuffle(data,labels)
    debug_msg = 'Data successfully shuffled'
    print(debug_msg)
    return shuffled_data, shuffled_labels

def PCA_fit(traindata,var_explained):
    pca = PCA(var_explained)
    pca.fit(traindata)
    return pca

def PCA_transform(pca, data):
    # Transform
    data_out = pca.transform(data)
    print(data_out.shape)
    return data_out
    
def Resample(data, labels, sampling_method='TomekLinks'):
    if sampling_method == 'SMOTETomek':
        rs = SMOTETomek(ratio='auto')
    elif sampling_method == 'TomekLinks':
        rs = TomekLinks(sampling_strategy='majority')
    elif sampling_method == 'InstanceHardnessThreshold':
        rs = InstanceHardnessThreshold(estimator=LogisticRegression(
                solver='lbfgs', multi_class='auto'))
    elif sampling_method == 'OneSidedSelection':
        rs = OneSidedSelection(sampling_strategy='majority')
    X_rs, y_rs = rs.fit_sample(data,labels)
    print("Resampling complete. Sample amount changed from "+str(data.shape[0])+' to '+str(X_rs.shape[0]))
    return X_rs, y_rs

def Low_variance(X,threshold):
    X = Remove_Zero_Variance(X)
    X_norm = scale(X,axis=1)
    X_var_norm = np.var(X_norm,axis=0)
    low_variances = np.where(X_var_norm < threshold)[0]
    return list(low_variances)