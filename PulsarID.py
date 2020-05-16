# Python Program to detect if star is Pulsar or not
# By: Justin Rodriguez
# 5/15/20

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pylab as plt

import random
import time

# for the pre-processing of the data
from sklearn.preprocessing import StandardScaler

# for machine learning models
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

# for the metrics of machine learning models
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.model_selection import KFold

pulsar_stars = pd.read_csv("Data\pulsar_stars.csv")  # Read the data

# Verify cleanness of data
print(pulsar_stars.apply(pd.Series.count))
print(pulsar_stars.head(10))
print(pulsar_stars.describe())

# Set see for random cursor
random.seed(10)

# Split data
X = pulsar_stars.drop(['target_class'], axis=1)
y = pulsar_stars['target_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=10)
X_columns = X_train.columns

# Correlation Plot
corr = pulsar_stars.corr()
sns.heatmap(corr)
plt.show()
# Create Numeric Plots
num = [f for f in X_train.columns if X_train.dtypes[f] != 'object']
nd = pd.melt(X_train, value_vars=num)
n1 = sns.FacetGrid(nd, col='variable', col_wrap=3, height=5.5, sharex=False, sharey=False)
n1 = n1.map(sns.distplot, 'value')
plt.show()
# Pre-Processing of Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X_columns)
X_test = pd.DataFrame(X_test, columns=X_columns)

# Train Model
columns = ['Decision_tree', 'Logistic_regression', 'Random_forest', 'K-NNeighbors', 'neural_network']
index = ['time(s)', 'accuracy', 'recall', 'f1_score_weighted', 'AUC']
performance_df = pd.DataFrame(columns=columns, index=index)

# Create DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=4)
dtc.fit(X_train, y_train)
params = {'max_depth': [2, 4, 8]}
dtcgrid = GridSearchCV(estimator=dtc, param_grid=params, cv=KFold(5, random_state=10), scoring='accuracy')
dtcgrid.fit(X_train, y_train)

# LogisticRegression
lrg = LogisticRegression(C=0.001, solver='liblinear')
lrg.fit(X_train, y_train)
params = {'C': [0.01, 0.1, 1, 10]}
lrggrid = GridSearchCV(estimator=lrg, param_grid=params, cv=KFold(5, random_state=10), scoring='accuracy')
lrggrid.fit(X_train, y_train)

# RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, max_depth=10)
rfc.fit(X_train, y_train)
params = {'n_estimators': [10, 20, 50, 100], 'max_depth': [10, 50]}
rfcgrid = GridSearchCV(estimator=rfc, param_grid=params, cv=KFold(5, random_state=10), scoring='accuracy')
rfcgrid.fit(X_train, y_train)

# KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=10, p=2)
knc.fit(X_train, y_train)
params = {'n_neighbors': [2, 5, 10, 50], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
kncgrid = GridSearchCV(estimator=knc, param_grid=params, cv=KFold(5, random_state=10), scoring='accuracy')
kncgrid.fit(X_train, y_train)

# Define Best Models
dtc_best = dtcgrid.best_estimator_
lrg_best = lrggrid.best_estimator_
rfc_best = rfcgrid.best_estimator_
knc_best = kncgrid.best_estimator_

# Run each model to examine time and performance
start = time.time()
dtc = dtc_best
dtc_best.fit(X_train, y_train)
end = time.time()
print('Time for Decision Tree Classifier = ', end - start, 's')
performance_df['Decision_tree']['time(s)'] = end - start

# Next LogisticRegression
start = time.time()
lrg = lrg_best
lrg_best.fit(X_train, y_train)
end = time.time()
print('Time for Logistic Regression = ', end - start, 's')
performance_df['Logistic_regression']['time(s)'] = end - start

# Then RandomForestClassifier
start = time.time()
rfc = rfc_best
rfc_best.fit(X_train, y_train)
end = time.time()
print('Time for Random Forest Classifier = ', end - start, 's')
performance_df['Random_forest']['time(s)'] = end - start

# Lastly KNeighborClassifier
start = time.time()
knc = knc_best
knc_best.fit(X_train, y_train)
end = time.time()
print('Time for K-Neighbors Classifier = ', end - start, 's')
performance_df['K-NNeighbors']['time(s)'] = end - start

# Calculate Scores For Each Model
y_predict_dtc = dtc_best.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_dtc)
recall = recall_score(y_test, y_predict_dtc)
performance_df['Decision_tree']['accuracy'] = accuracy
performance_df['Decision_tree']['recall'] = recall

y_predict_lrg = lrg_best.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_lrg)
recall = recall_score(y_test, y_predict_lrg)
performance_df['Logistic_regression']['accuracy'] = accuracy
performance_df['Logistic_regression']['recall'] = recall

y_predict_rfc = rfc_best.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_rfc)
recall = recall_score(y_test, y_predict_rfc)
performance_df['Random_forest']['accuracy'] = accuracy
performance_df['Random_forest']['recall'] = recall

y_predict_knc = knc_best.predict(X_test)
accuracy = accuracy_score(y_test, y_predict_knc)
recall = recall_score(y_test, y_predict_knc)
performance_df['K-NNeighbors']['accuracy'] = accuracy
performance_df['K-NNeighbors']['recall'] = recall

# Confusion Matrix for Decision Tree Classifier
dtc_confm = confusion_matrix(y_test, y_predict_dtc)
dtc_df = pd.DataFrame(dtc_confm)
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(dtc_df.T, annot=True, annot_kws={"size": 15}, cmap='Oranges', vmin=0, vmax=800, fmt='.0f',
            linewidths=1, linecolor="white", cbar=False, xticklabels=["no pulsar star", "pulsar star"],
            yticklabels=["no pulsar star", "pulsar star"])
plt.ylabel("Predicted", fontsize=15)
plt.xlabel("Actual", fontsize=15)
ax.set_xticklabels(["no pulsar star", "pulsar star"], fontsize=13)
ax.set_yticklabels(["no pulsar star", "pulsar star"], fontsize=13)
plt.title("Confusion Matrix for Decision Tree Classifier", fontsize=15)
plt.show()
print("")
print(classification_report(y_test, y_predict_dtc))
performance_df['Decision_tree']['f1_score_weighted'] = f1_score(y_test, y_predict_dtc, average='weighted')

# Confusion Matrix for Logistic Regression Classifier
lrg_confm = confusion_matrix(y_test, y_predict_lrg)
lrg_df = pd.DataFrame(lrg_confm)
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(lrg_df.T, annot=True, annot_kws={"size": 15}, cmap='Purples', vmin=0, vmax=800, fmt='.0f',
            linewidths=1, linecolor="white", cbar=False, xticklabels=["no pulsar star", "pulsar star"],
            yticklabels=["no pulsar star", "pulsar star"])
plt.ylabel("Predicted", fontsize=15)
plt.xlabel("Actual", fontsize=15)
ax.set_xticklabels(["no pulsar star", "pulsar star"], fontsize=13)
ax.set_yticklabels(["no pulsar star", "pulsar star"], fontsize=13)
plt.title("Confusion Matrix for Logistic Regression Classifier", fontsize=15)
plt.show()
print("")
print(classification_report(y_test, y_predict_lrg))
performance_df['Logistic_regression']['f1_score_weighted'] = f1_score(y_test, y_predict_lrg, average='weighted')

# Confusion Matrix for Random Forest Classifier
rfc_confm = confusion_matrix(y_test, y_predict_rfc)
rfc_df = pd.DataFrame(rfc_confm)
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(rfc_df.T, annot=True, annot_kws={"size": 15}, cmap='Greens', vmin=0, vmax=800, fmt='.0f',
            linewidths=1, linecolor="white", cbar=False, xticklabels=["no pulsar star", "pulsar star"],
            yticklabels=["no pulsar star", "pulsar star"])
plt.ylabel("Predicted", fontsize=15)
plt.xlabel("Actual", fontsize=15)
ax.set_xticklabels(["no pulsar star", "pulsar star"], fontsize=13)
ax.set_yticklabels(["no pulsar star", "pulsar star"], fontsize=13)
plt.title("Confusion Matrix for Random Forest Classifier", fontsize=15)
plt.show()
print("")
print(classification_report(y_test, y_predict_rfc))
performance_df['Random_forest']['f1_score_weighted'] = f1_score(y_test, y_predict_rfc, average='weighted')

# Confusion Matrix for K-NNeighbor Classifier
knc_confm = confusion_matrix(y_test, y_predict_knc)
knc_df = pd.DataFrame(knc_confm)
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(knc_df.T, annot=True, annot_kws={"size": 15}, cmap='gray', vmin=0, vmax=800, fmt='.0f',
            linewidths=1, linecolor="white", cbar=False, xticklabels=["no pulsar star", "pulsar star"],
            yticklabels=["no pulsar star", "pulsar star"])
plt.ylabel("Predicted", fontsize=15)
plt.xlabel("Actual", fontsize=15)
ax.set_xticklabels(["no pulsar star", "pulsar star"], fontsize=13)
ax.set_yticklabels(["no pulsar star", "pulsar star"], fontsize=13)
plt.title("Confusion Matrix for K-NNeighbor Classifier", fontsize=15)
plt.show()
print("")
print(classification_report(y_test, y_predict_knc))
performance_df['K-NNeighbors']['f1_score_weighted'] = f1_score(y_test, y_predict_knc, average='weighted')

# ROC Curves for Classifiers

# Decision Tree Model ROC
dtc_best_prob = dtc_best.predict_proba(X_test)
fpr_logis, tpr_logis, thresholds_logis = roc_curve(y_test, dtc_best_prob[:, 1])
fig, ax = plt.subplots(figsize=(10, 7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_logis, tpr_logis)
plt.fill_between(fpr_logis, tpr_logis, alpha=0.2, color='b')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
AUC = roc_auc_score(y_test, dtc_best_prob[:, 1])
plt.title('Decision Tree Classifier ROC curve: AUC={0:0.3f}'.format(AUC))
plt.show()
performance_df['Decision_tree']['AUC'] = AUC

# Logistic Model ROC
lrg_best_prob = lrg_best.predict_proba(X_test)
fpr_logis, tpr_logis, thresholds_logis = roc_curve(y_test, lrg_best_prob[:, 1])

fig, ax = plt.subplots(figsize=(10, 7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_logis, tpr_logis)
plt.fill_between(fpr_logis, tpr_logis, alpha=0.2, color='b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
AUC = roc_auc_score(y_test, lrg_best_prob[:, 1])
plt.title('Logistic Regression ROC curve: AUC={0:0.3f}'.format(AUC))
plt.show()

performance_df['Logistic_regression']['AUC'] = AUC

# Random Forest Model ROC
rfc_best_prob = rfc_best.predict_proba(X_test)
fpr_logis, tpr_logis, thresholds_logis = roc_curve(y_test, rfc_best_prob[:, 1])

fig, ax = plt.subplots(figsize=(10, 7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_logis, tpr_logis)
plt.fill_between(fpr_logis, tpr_logis, alpha=0.2, color='b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
AUC = roc_auc_score(y_test, rfc_best_prob[:, 1])
plt.title('Random Forest ROC curve: AUC={0:0.3f}'.format(AUC))
plt.show()
performance_df['Random_forest']['AUC'] = AUC

# KNN Model ROC
knc_best_prob = knc_best.predict_proba(X_test)
fpr_logis, tpr_logis, thresholds_logis = roc_curve(y_test, knc_best_prob[:, 1])

fig, ax = plt.subplots(figsize=(10, 7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_logis, tpr_logis)
plt.fill_between(fpr_logis, tpr_logis, alpha=0.2, color='b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
AUC = roc_auc_score(y_test, knc_best_prob[:, 1])
plt.title('K_Neighbors ROC curve: AUC={0:0.3f}'.format(AUC))
plt.show()

performance_df['K-NNeighbors']['AUC'] = AUC


# Logistic regression is the best model
# Now we want to see the importance of each feature on the result
df_importance = pd.DataFrame()
columns = X_train.columns
importances = np.abs(lrg_best.coef_[0])

for i in range(len(columns)):
    df_importance[columns[i]] = [importances[i]]

df_importance.insert(0, '', 'Important features')
df_importance.head(10)
