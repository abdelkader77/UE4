import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from numpy import pi
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

def PrepareTarget():
    histologies= dataFrame[sheetColumns[6]];
    target=[];
    unspecified=[];
    index=0;
    for histology in histologies:
        histologyLower= histology.lower();
        if "squamous" in histologyLower:
            target.append([1]);
        elif "adenocarcinoma" in histologyLower or "papillary" in histologyLower or "micropapillary" in histologyLower or "mucinous" in histologyLower or "acinar" in histologyLower or "solid"in histologyLower:
            target.append([0]);
        else:
            unspecified.append(index);
        index=index+1;
    target= np.array(target);
    return target,unspecified;

dataFrame = pd.read_excel('Data/Lung3.metadata.xls', sheet_name='Lung3.metadata')
sheetColumns= dataFrame.columns;
dataFrameNameIndexed= dataFrame.set_index(sheetColumns[0]);
sampleSerie= dataFrame[sheetColumns[0]];
sampleList= sampleSerie.tolist();

targetArray,unspecifiedPatients= PrepareTarget();
inputFrame = pd.read_csv('Data/GSE58661_RAW/Expressions.csv',sep=';');
featureArray= inputFrame.values;
featureArray= featureArray.transpose();
featureIndices = np.arange(featureArray.shape[-1]);
featureNames= featureArray[0,:];
featureArray= featureArray[1:,:];
featureArray= np.delete(featureArray, unspecifiedPatients, axis=0);
targetArray= targetArray.ravel();

selector = SelectPercentile(f_classif, percentile=1000/60607);
selector.fit(featureArray, targetArray);
scores = -np.log10(selector.pvalues_);
scores /= scores.max();
featureIndices=featureIndices.reshape(1,-1);
selectedFeatureIndices=selector.transform(featureIndices);
selectedFeatureNames=featureNames[selectedFeatureIndices];
scores=scores.reshape(1,-1);
selectedScores= selector.transform(scores);

testRatio = (100-70)/100;
featureArrayTrain, featureArrayTest, targetArrayTrain, targetArrayTest = train_test_split(selector.transform(featureArray), targetArray, test_size=testRatio,random_state=0);
decisionTree = tree.DecisionTreeClassifier();
decisionTree = decisionTree.fit(selector.transform(featureArray), targetArray);
predictionArray= decisionTree.predict(featureArrayTest);
falsePositiveRate, truePositiveRate, _ = roc_curve(targetArrayTest, predictionArray)
rocAreaUnderCover = auc(falsePositiveRate, truePositiveRate);
confusionMatrix = confusion_matrix(targetArrayTest, predictionArray);

plt.figure()
plt.plot(falsePositiveRate, truePositiveRate, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % rocAreaUnderCover);
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic example');
plt.legend(loc="lower right");
plt.show();

dotData = tree.export_graphviz(decisionTree, out_file=None, 
                         feature_names=selectedFeatureNames.transpose(),  
                         class_names=np.array(["Adenocarcinoma","Squamous Cell Carcinoma"]),  
                         filled=True, rounded=True,  
                         special_characters=True);
graph = graphviz.Source(dotData); 
graph.view("Decision Tree 1000");
