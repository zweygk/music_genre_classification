from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier


def Learn_Multiclass_SVM(data,labels):
    model = OneVsRestClassifier(SVC())
    
    parameters = {
    "estimator__C": [2**6,2**7,2**8,2**9,2**10,2**11,2**12.2**14],
    "estimator__kernel": ["rbf"],
    "estimator__gamma":[2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10]
    }
    
    accuracy = make_scorer(accuracy_score)
    
    model_tuned = GridSearchCV(model, param_grid=parameters, scoring=accuracy, cv=6)
    model_tuned.fit(data, labels)
    
    best_score = model_tuned.best_score_
    best_params = model_tuned.best_params_
    
    debug_msg = 'Best score: '+str(best_score)+', using parameters: '+str(best_params)
    print(debug_msg)
    return model_tuned

def Learn_Random_Forest(data, labels):
    model = OneVsRestClassifier(RandomForestClassifier())
    parameters = {'estimator__max_depth': [7,8,10,11,12,13]}
    model_tuned = GridSearchCV(model, parameters)
    model_tuned.fit(data, labels)
    
    best_score = model_tuned.best_score_
    best_params = model_tuned.best_params_
    
    debug_msg = 'Best score: '+str(best_score)+', using parameters: '+str(best_params)
    print(debug_msg)

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
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import regularizers

def Learn_Multilayer_Perceptron(data, labels, learning_rate=0.0001,validation_split=0.33):
    labels = labels-1
    categorical_labels = to_categorical(labels,num_classes=10)
    model = Sequential()
    model.add(Dense(64,activation='relu',
                    kernel_regularizer=regularizers.l2(0.0001),
                    input_dim=data.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu',
                    kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    
    filepath = 'best_perceptron_model.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    model.fit(data, categorical_labels,validation_split=0.33,batch_size=512, epochs=50000, callbacks=callbacks_list)
    return model
    
def Evaluate_NN_Performance(model_name, test_data, test_labels):
    model = load_model(model_name)
    test_labels = test_labels - 1
    categorical_labels = to_categorical(test_labels, num_classes=10)
    score = model.evaluate(test_data, categorical_labels, batch_size=512)
    debug_msg = 'Accuracy score of Neural Network: ' + str(score)
    print(debug_msg)
    return score
    
    
    
    
    
    
    