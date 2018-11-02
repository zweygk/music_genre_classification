from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import cross_val_score




def Learn_Multiclass_SVM(data,labels,gamma_exps=[4], C_exps=[6.5]):
    model = OneVsRestClassifier(SVC(class_weight='auto',))
    parameters = {
    "estimator__C": list(np.float_power(3,C_exps)),
    "estimator__kernel": ["rbf"],
    "estimator__gamma": list(np.float_power(2,gamma_exps))
    }
    
    accuracy = make_scorer(accuracy_score)
    
    model_tuned = GridSearchCV(model, param_grid=parameters, scoring=accuracy, cv=5,verbose=True)
    model_tuned.fit(data, labels)
    
    best_score = model_tuned.best_score_
    best_params = model_tuned.best_params_
    
    debug_msg = 'Best score: '+str(best_score)+', using parameters: '+str(best_params)
    print(debug_msg)
    return model_tuned

def Learn_SVM(X_train,y_train):
    model = GridSearchCV(LinearSVC(penalty='l2', loss='squared_hinge', dual=False,
                       tol=1e-4,multi_class='ovr'), param_grid={"C":[0.001,0.01,0.1]},cv=5,verbose=True)
    model.fit(X_train,y_train)
    
    best_score = model.best_score_
    best_params = model.best_params_
    
    debug_msg = 'Best score: '+str(best_score)+', using parameters: '+str(best_params)
    print(debug_msg)
    return model
    
# NUMPY ARRAYS AS INPUT!
def Learn_XGBoost(train_features,train_labels,test_features,test_labels,max_depth=500,num_round=100,use_weights=False,verbose=False):
    model_name = 'xboost.mdl'
    train_labels = train_labels-1 # Get labels 0 - 9
    test_labels = test_labels-1 # Get labels 0 - 9
    train_labels.resize(train_labels.shape[0])
    test_labels.resize(test_labels.shape[0])
    if(use_weights):        
        class_weights = class_weight.compute_class_weight('balanced'
                                               ,np.unique(train_labels)
                                               ,train_labels)
        train_weights = np.zeros(train_labels.shape[0])
        test_weights = np.zeros(test_labels.shape[0])
        for i in range(train_labels.shape[0]):
            train_weights[i] = class_weights[train_labels[i]]
        for i in range(test_labels.shape[0]):
            test_weights[i] = class_weights[test_labels[i]]
    else:
        train_weights = None
        test_weights = None
    dtrain = xgb.DMatrix(data=train_features,label=train_labels,weight=train_weights)
    dtest = xgb.DMatrix(data=test_features,label=test_labels,weight=test_weights)
    param = {'max_depth':max_depth, 'eta':1, 'silent':1, 'objective':'multi:softprob', 'num_class':10}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')] 
    bst = xgb.train(param,dtrain,num_round,watchlist,verbose_eval=verbose)
    bst.save_model(model_name)
    return bst

# Returns probabilities for each label from every prediction
def XGB_Predict(xgb_model,test_features,test_labels):
    test_labels=test_labels-1 # Transform to range 0-9
    dtest = xgb.DMatrix(data=test_features,label=test_labels)
    pred_labels_score = xgb_model.predict(dtest)
    pred_labels = np.argmax(pred_labels_score,axis=1)
    pred_labels = pred_labels + 1 # Transform back to range 1-10
    return pred_labels, pred_labels_score

def Learn_Random_Forest(data, labels):
    model = OneVsRestClassifier(RandomForestClassifier(n_estimators=500,
                                                       criterion='entropy',
                                                       oob_score=True,verbose=True))
    parameters = {'estimator__max_depth': [13]}
    model_tuned = GridSearchCV(model, parameters)
    model_tuned.fit(data, labels)
    
    best_score = model_tuned.best_score_
    best_params = model_tuned.best_params_
    
    debug_msg = 'Best score: '+str(best_score)+', using parameters: '+str(best_params)
    print(debug_msg)
    return model_tuned

def Predict(model, test_data):
    predictions = model.predict(test_data)
    return predictions

def Accuracy_Score(true, predictions):
    score = accuracy_score(true, predictions)
    debug_msg = 'Accuracy: '+str(score)
    print(debug_msg)
    return score


from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import datetime

def Create_Multilayer_Perceptron(data, labels):
    model = Sequential()
    model.add(Dense(units=128, input_dim=data.shape[1],
                    activation = 'sigmoid',
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.53))
    #model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.53))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.53))
    model.add(Dense(10, activation='softmax'))
    return model

def Learn_Multilayer_Perceptron(data, labels, model, learning_rate=0.0001,validation_split=0.33):
    labels = labels-1
    categorical_labels = to_categorical(labels,num_classes=10)
    
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    
    callbacks_list, save_path = Callbacks()
    
    model.fit(data, categorical_labels,validation_split=0.33,batch_size=32, epochs=50000, callbacks=callbacks_list)
    model = load_model(save_path)
    
    return model

def Callbacks():
    start_datetime = str(datetime.datetime.now())
    for ch in [' ',':','.']:
        start_datetime = start_datetime.replace(ch,'_')
    save_path = 'best_perceptron_model_'+start_datetime+'.h5'
    checkpoint = ModelCheckpoint(save_path, monitor='val_acc',
                                 verbose = 1, save_best_only=True,
                                 mode='max',period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01,
                                   patience=1200, verbose=1, mode='max')
    callbacks_list = [checkpoint,early_stopping]
    return callbacks_list, save_path
    
    
def Evaluate_NN_Performance(model, test_data, test_labels):
    test_labels = test_labels - 1
    categorical_labels = to_categorical(test_labels, num_classes=10)
    score = model.evaluate(test_data, categorical_labels, batch_size=512)
    debug_msg = 'Accuracy score of Neural Network: ' + str(score)
    print(debug_msg)
    return score
    
    
    
    
    
    
    