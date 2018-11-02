import seaborn as sns
from matplotlib import pyplot as plt
import data_utils as du
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score,accuracy_score,balanced_accuracy_score,classification_report
from sklearn import preprocessing
sns.set()

def hist(values,label="Values",title="Title"):
    df = pd.DataFrame(data=values)
    plt.figure()
    ax = sns.distplot(df, bins=10, kde=False, rug=False,axlabel=label);
    ax.xaxis.set_major_locator(plt.MaxNLocator(10));
    plt.title(title,fontsize=20);
    
# Labels: np.array
def label_distribution(label_array,size=(10,5)):
    labels = pd.DataFrame(data=label_array)
    plt.figure(figsize=size)
    ax = sns.distplot(labels, bins=10, kde=False, rug=False,axlabel="Labels");
    ax.xaxis.set_major_locator(plt.MaxNLocator(10));
    plt.title("Distribution of labels",fontsize=20);
    label = 1
    print("Frequency of labels in percentage:")
    for freq_i,n_i in zip(labels[0].value_counts()/labels.shape[0],labels[0].value_counts()):
        print("Label {} : {} samples, frequency: {:.3f} %".format(label,n_i,freq_i*100))
        label += 1


# Variances of features
def feature_variances(X):
    #print("Removing zero variance features: ")
    # 0. Remove 0 -variance features
    X = du.Remove_Zero_Variance(X)
    # 1. Normalize data
    #X_norm = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
    X_norm = preprocessing.scale(X,axis=1)
    # 2. Calculate variance for each feature
    X_var_norm = np.var(X_norm,axis=0);
    # 3. Visualize variance
    f, ax = plt.subplots(figsize=(30, 8));
    #sns.set(color_codes=True);
    sns.distplot(X_var_norm,kde=False,rug=False,bins='auto');
    ax.set(xlabel='Variance', ylabel='Number of features');
    ax.set_title("Variance across features");
    for p in range(9):
        perc_i = (p + 1)*10
        print(perc_i,":th percentile: ",np.percentile(X_var_norm,perc_i))

# Input: pandas dataframe
def feature_correlations(data,corr_threshold=0.995,perc=99):
    X = du.Remove_Zero_Variance(data)
    data_df = pd.DataFrame(data=X)
    corr = data_df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 18))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
   
    # Analyze distribution of correlation coefficients
    cor_m = corr.values
    # Remove 1s from diagonal
    for i in range(cor_m.shape[0]):
        cor_m[i,i] = -10
    print("Largest correlation: ",cor_m.max())
    indeces = np.where(cor_m >= corr_threshold)
    print("Indeces with stronger correlance than ",corr_threshold,":\n",indeces[0],"\n",indeces[1])
    print("Correlation of percentile ",perc,": ",np.percentile(cor_m[0,:],perc))
    
    
# pred_labels : [n_labels] The predicted labels
# pred_labels_score : [n_labels,n_classes] probability score of each class for each prediction
def ROC_curve(test_labels,pred_labels_score, fig_size=(15,15)):
    test_labels = test_labels - 1 # Transform to range 0-9
    test_labels_bin = label_binarize(test_labels, classes=range(10))
    roc_auc = dict()
    fpr = dict()
    tpr = dict()
    n_classes = test_labels_bin.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:,i], pred_labels_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #Plot ROC curve
    plt.figure(figsize=fig_size)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i+1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC')
    plt.legend(loc="lower right")
    plt.show()

   
def performance_report(test_labels,pred_labels):
    print(classification_report(test_labels,pred_labels))
    print("Accuracy: ",accuracy_score(test_labels,pred_labels))
    print("Balanced Accuracy: ",balanced_accuracy_score(test_labels,pred_labels))

def plot_confusion_mat(npmatrix,size):
    df = pd.DataFrame(npmatrix)
    plt.figure(figsize = (size[0],size[1]))
    heatmap = sns.heatmap(df)
    