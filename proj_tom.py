import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sys import maxsize
pd.set_printoptions(threshold=maxsize)
print('#','~'*15,'Load Data','~'*50 )
od_one = pd.read_table("/Users/thomaskeeley/Documents/school/data_mining/occupancy_data/datatest.txt", sep=',')
od_two = pd.read_table("/Users/thomaskeeley/Documents/school/data_mining/occupancy_data/datatraining.txt", sep=',')
od_three = pd.read_table("/Users/thomaskeeley/Documents/school/data_mining/occupancy_data/datatest2.txt", sep=',')

print(list(od_one))
print(list(od_two))
print(list(od_three))

#Combine datasets
od_full = pd.concat([od_one,od_two,od_three])

#Split the DateTime feature to separate columns
od_full[['date','time']] = od_full.date.apply(lambda x: pd.Series(str(x).split(" ")))
od_full.head()

#Create new feature for work hours (6am-6pm) labeling as binary
od_full['WorkHours'] = np.where((od_full['time'] >= '06:00:00') & (od_full['time'] <= '18:00:00'), '1', '0')
od_full.head()

#Change the order of the columns for easier processing later
od_full = od_full[['Temperature', 'CO2', 'Humidity', 'HumidityRatio', 'Light', 'WorkHours', 'Occupancy']]
od_full.to_csv("/Users/thomaskeeley/Documents/school/occupancy-data.csv")

print("**columns**")
print(list(od_full))
print(od_full.head())

print("\n**dimensions**")
print(od_full.shape)

print("\n**info**")
print(od_full.info())

print("\n**missings??**")
print(od_full.isna().sum())

print("\n**stats**")
print(od_full.describe)

#check distribution
f, ax = plt.subplots(figsize=(11,11))
sns.pairplot(od_full)
plt.show()
#Skewness present and possibility of colinearity. May need to use regularization in models

#Create a corr plot to further visualize the relationship of the features
#High correlation between humidity and humidity ratio
corr = od_full.corr()
print(corr)

sns.heatmap(corr, cmap='Blues', center=0, square=True, linewidths=.5, annot=True)
plt.title('Correlation Plot', fontsize=10)
plt.tight_layout()
plt.show()


print("\nSkewness (normal is 0):\n")
print(od_full.skew()) #skewness in Light and CO2

########################################################################
#%%                           PreProcessing
########################################################################
print('#','~'*15,'Preprocessing','~'*50 )
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
class_le = LabelEncoder()
scaler = StandardScaler()
normer = Normalizer()
od_full['Occupancy'] = class_le.fit_transform(od_full['Occupancy'])
feats = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'WorkHours']
target = ['Occupancy']
X = od_full[feats]
y = od_full[target]

#normalize features to be on same scale
X = normer.fit_transform(X)
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=feats)

print("\n**skewness after transformation**")#skewness improves everywhere except temperature, where it slightly worsens
print(X.skew())

#make train & test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

print("\n**Train dim**")
print(X_train.shape, y_train.shape)

print("\n**Test dim**")
print(X_test.shape, y_test.shape)
print('#','~'*15,'Modeling Process','~'*50 )

#Modeling
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import VotingClassifier
from pydotplus import graph_from_dot_data


########################################################################
#%%                         Logistic Regression
########################################################################

print('#','~'*10,'Logistic Regression','~'*10 )
lr_clf = LogisticRegression()
lr = lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_test)

#Logistic Regression performance
print("\n**Logistic Regression Accuracy Score**")
print(accuracy_score(y_test, y_pred_lr))

print("\n**Logistic Regression Classification Report**")
print(classification_report(y_test,y_pred_lr))

print("\n**Logistic Regression Confusion Matrix**")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

sns.heatmap(cm_lr, annot=True, fmt='d', cbar=True, cmap='Blues', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - Logistic Regression', fontsize=30)
plt.show()

#ROC LR
fpr, tpr, _ = roc_curve(y_test, y_pred_lr)
auc = roc_auc_score(y_test, y_pred_lr)
plt.plot(fpr,tpr,label="Decision Tree, auc="+str(auc))
plt.legend(loc=4)
plt.show()


########################################################################
#%%                           Decision Tree
########################################################################
print('#','~'*10,'Decision Tree','~'*10 )
dt_clf = DecisionTreeClassifier(criterion='gini')
dt = dt_clf.fit(X_train,y_train)
y_pred_dt = dt_clf.predict(X_test)

#DT performance
print("\n**Decision Tree Accuracy Score**")
print(accuracy_score(y_test, y_pred_dt))

print("\n**Decision Tree Classification Report**")
print(classification_report(y_test,y_pred_dt))

print("\n**Decision Tree Confusion Matrix**")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

sns.heatmap(cm_dt, annot=True, fmt='d', cbar=True, cmap='BuGn', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - Decision Tree', fontsize=30)
plt.show()

#ROC DT
fpr, tpr, _ = roc_curve(y_test, y_pred_dt)
auc = roc_auc_score(y_test, y_pred_dt)
plt.plot(fpr,tpr,label="Decision Tree, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#DT plot
dot_data = export_graphviz(dt, filled=True)
graph = graph_from_dot_data(dot_data)
graph.write_pdf("/Users/thomaskeeley/Documents/decision_tree.pdf")
webbrowser.open_new(r'decision_tree.pdf')

#DT feature importance
importances = dt_clf.feature_importances_
f_importances = pd.Series(importances, od_full.iloc[:, 1:7].columns)
f_importances.sort_values(ascending=False, inplace=True)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(15, 12), rot=90, fontsize=20)
plt.ylabel('Importance', fontsize=30)
plt.xlabel('Feature', fontsize=30)
plt.title('Feature Importance in Decision Tree', fontsize=30)
plt.tight_layout()
plt.show()

########################################################################
#%%                           Random Forest
########################################################################
#Random forest
print('#','~'*10,'Random Forest','~'*10 )
rf_clf = RandomForestClassifier(n_estimators=100)
rf = rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

#RF performance
print("\n**Random Forest Accuracy Score**")
print(accuracy_score(y_test, y_pred_rf))

print("\n**Random Forest Classification Report**")
print(classification_report(y_test,y_pred_rf))

print("\n**Random Forest Confusion Matrix**")
print(confusion_matrix(y_test,y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.heatmap(cm_rf, annot=True, fmt='d', cbar=True, cmap='Blues', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - Random Forest', fontsize=30)
plt.show()

#ROC RF
fpr, tpr, _ = roc_curve(y_test, y_pred_rf)
auc = roc_auc_score(y_test, y_pred_rf)
plt.plot(fpr,tpr,label="Random Forest, auc="+str(auc))
plt.legend(loc=4)
plt.show()

########################################################################
#%%                               SVM
########################################################################

#SVM
print('#','~'*10,'SVM','~'*10 )
svm_clf = svm.SVC(kernel="poly", gamma=2)
svmachine = svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

#SVM performance
print("\n**SVM Accuracy Score**")
print(accuracy_score(y_test, y_pred_svm))

print("\n**SVM Classification Report**")
print(classification_report(y_test,y_pred_svm))

print("\n**SVM Confusion Matrix**")
print(confusion_matrix(y_test,y_pred_svm))
cm_svm = confusion_matrix(y_test,y_pred_svm)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.heatmap(cm_svm, annot=True, fmt='d', cbar=True, cmap='Blues', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - SVM', fontsize=30)
plt.show()

#ROC SVM
fpr, tpr, _ = roc_curve(y_test, y_pred_svm)
auc = roc_auc_score(y_test, y_pred_svm)
plt.plot(fpr,tpr,label="SVM, auc="+str(auc))
plt.legend(loc=4)
plt.show()

########################################################################
#%%                               KNN
########################################################################

#KNN
print('#','~'*10,'KNN','~'*10 )
knn_clf = KNeighborsClassifier(n_neighbors= 5)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)

#KNN performance
print("\n**KNN Accuracy Score**")
print(accuracy_score(y_test, y_pred_knn))

print("\n**KNN Classification Report**")
print(classification_report(y_test,y_pred_knn))

print("\n**KNN Confusion Matrix**")
print(confusion_matrix(y_test,y_pred_knn))
cm_knn = confusion_matrix(y_test,y_pred_knn)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.heatmap(cm_knn, annot=True, fmt='d', cbar=True, cmap='Blues', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - KNN', fontsize=30)
plt.show()

#ROC KNN
fpr, tpr, _ = roc_curve(y_test, y_pred_knn)
auc = roc_auc_score(y_test, y_pred_svm)
plt.plot(fpr,tpr,label="KNN, auc="+str(auc))
plt.legend(loc=4)
plt.show()

########################################################################
#%%                               XGBoost
########################################################################
#Grid Search for hyperparameter tuning
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8, 10, 12]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=111)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
# summarize results
print(grid_result.best_score_, grid_result.best_params_)

# XGBoost
print('#','~'*10,'XGBoost','~'*10 )
xgb_clf = XGBClassifier(max_depth= 12, n_estimators=100)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

#XGBoost performance
print("\n**XGBoost Accuracy Score**")
print(accuracy_score(y_test, y_pred_xgb))

print("\n**XGBoost Classification Report**")
print(classification_report(y_test,y_pred_xgb))

print("\n**XGBoost Confusion Matrix**")
print(confusion_matrix(y_test,y_pred_xgb))
cm_xgb = confusion_matrix(y_test,y_pred_xgb)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.heatmap(cm_xgb, annot=True, fmt='d', cbar=True, cmap='Blues', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - XGBoost', fontsize=30)
plt.show()

#ROC XGBoost
fpr, tpr, _ = roc_curve(y_test, y_pred_xgb)
auc = roc_auc_score(y_test, y_pred_xgb)
plt.plot(fpr,tpr,label="KNN, auc="+str(auc))
plt.legend(loc=4)
plt.show()


########################################################################
#%%                           AdaBoost
########################################################################

# AdaBoost
print('#','~'*10,'AdaBoost','~'*10 )
adb_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=12), n_estimators=100, random_state=111)
adb_clf.fit(X_train, y_train)
y_pred_adb = adb_clf.predict(X_test)

#AdaBoost performance
print("\n**AdaBoost Accuracy Score**")
print(accuracy_score(y_test, y_pred_adb))

print("\n**AdaBoost Classification Report**")
print(classification_report(y_test,y_pred_adb))

print("\n**AdaBoost Confusion Matrix**")
print(confusion_matrix(y_test,y_pred_adb))
cm_adb = confusion_matrix(y_test,y_pred_adb)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.heatmap(cm_adb, annot=True, fmt='d', cbar=True, cmap='Blues', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - AdaBoost', fontsize=30)
plt.show()

#ROC AdaBoost
fpr, tpr, _ = roc_curve(y_test, y_pred_adb)
auc = roc_auc_score(y_test, y_pred_adb)
plt.plot(fpr,tpr,label="KNN, auc="+str(auc))
plt.legend(loc=4)
plt.show()

########################################################################
#%%                           Blending Ensemble
########################################################################

#Create new train test split, further split train into validation set
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=111)
X_train_2, X_val, y_train_2, y_val = train_test_split(X_train2, y_train2, test_size=0.2, random_state=111)

#Fit new decision tree model
dt_model = DecisionTreeClassifier(criterion='gini')
dt_model.fit(X_train_2, y_train_2)
dt_val_pred = dt_model.predict(X_val)
dt_test_pred = dt_model.predict(X_test2)
dt_val_pred = pd.DataFrame(dt_val_pred)
dt_test_pred = pd.DataFrame(dt_test_pred)

#Fit new random forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_2, y_train_2)
rf_val_pred = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test2)
rf_val_pred = pd.DataFrame(rf_val_pred)
rf_test_pred = pd.DataFrame(rf_test_pred)

#Fit new SVM model
svm_model = svm.SVC(kernel="poly", gamma=2)
svm_model.fit(X_train_2, y_train_2)
svm_val_pred = svm_model.predict(X_val)
svm_test_pred = svm_model.predict(X_test2)
svm_val_pred = pd.DataFrame(svm_val_pred)
svm_test_pred = pd.DataFrame(svm_test_pred)

#Fit new KNN model
knn_model = KNeighborsClassifier(n_neighbors= 5)
knn_model.fit(X_train_2, y_train_2)
knn_val_pred = knn_model.predict(X_val)
knn_test_pred = knn_model.predict(X_test2)
knn_val_pred = pd.DataFrame(knn_val_pred)
knn_test_pred = pd.DataFrame(knn_test_pred)


#Reset index of validation and test set
X_val = X_val.reset_index()
X_test2 = X_test.reset_index()

#Combine the predictions of all three models
df_val = pd.concat([X_val, dt_val_pred, rf_val_pred, svm_val_pred, knn_val_pred], axis=1)
df_test = pd.concat([X_test2, dt_test_pred, rf_test_pred, svm_test_pred, knn_test_pred], axis=1)

#logistic regression on the blending of three models
print('#','~'*10,'Logistic Regression','~'*10 )
blend_lr = LogisticRegression()
blend_lr.fit(df_val, y_val)
y_pred_lr = blend_lr.predict(df_test)

#logistic regression performance
print("\n**Logistic Regression Accuracy Score**")
print(accuracy_score(y_test, y_pred_lr))

print("\n**Logistic Regression Classification Report**")
print(classification_report(y_test,y_pred_lr))

print("\n**Logistic Regression Confusion Matrix**")
print(confusion_matrix(y_test,y_pred_lr))
cm_lr = confusion_matrix(y_test,y_pred_lr)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.heatmap(cm_lr, annot=True, fmt='d', cbar=True, cmap='terrain', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - Blending Logistic Regression', fontsize=30)
plt.show()

#ROC logistic regression
fpr, tpr, _ = roc_curve(y_test, y_pred_lr)
auc = roc_auc_score(y_test, y_pred_lr)
plt.plot(fpr,tpr,label="Logistic Regression, auc="+str(auc))
plt.legend(loc=4)
plt.show()


########################################################################
#%%                        Hard Voting Ensemble
########################################################################

#Hard Voting (logistic regression, decision tree, random forest, SVM, KNN)
print('#','~'*10,'Hard Voting','~'*10 )
hard_voting = VotingClassifier(estimators=[('lr', lr_clf), ('dt', dt_clf), ('rf', rf_clf), ('svm', svm_clf), ('knn', knn_clf)], voting='hard')
hard_voting.fit(X_train, y_train)
hard_voting_pred = hard_voting.predict(X_test)

#Hard Voting performance
print("\n**Hard Voting Accuracy Score**")
print(accuracy_score(y_test, hard_voting_pred))

print("\n**Hard Voting Classification Report**")
print(classification_report(y_test,hard_voting_pred))

print("\n**Hard Voting Confusion Matrix**")
print(confusion_matrix(y_test,hard_voting_pred))
cm_hv = confusion_matrix(y_test,hard_voting_pred)

#plot confusison matrix
plt.figure(figsize=(12,10))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.heatmap(cm_hv, annot=True, fmt='d', cbar=True, cmap='BuGn', annot_kws={'size': 25})
plt.ylabel('True Label', fontsize=25)
plt.xlabel('Predicted Label', fontsize=25)
plt.title('Confusion Matrix - Hard Voting', fontsize=30)
plt.show()