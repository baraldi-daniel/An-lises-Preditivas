import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import haversine as hs
from haversine import Unit

#read and treat data
data = pd.read_csv(r'C:\Users\baral\OneDrive\Documents\Engenharia de Dados - USP\Análises Preditivas\Trabalho\dataset\fraudTrain.csv')

data = data.dropna()

print(data['is_fraud'].value_counts())

count_no_fraud = len(data[data['is_fraud']==0])
count_fraud = len(data[data['is_fraud']==1])
pct_of_no_fraud = count_no_fraud/(count_no_fraud+count_fraud)
print("percentage of no fraud is", pct_of_no_fraud*100)
pct_of_fraud = count_fraud/(count_no_fraud+count_fraud)
print("percentage of fraud", pct_of_fraud*100)


print(data['is_fraud'].value_counts())

graph=sns.countplot(x='is_fraud', data=data, palette='hls')
plt.ticklabel_format(style='plain', axis='y')
for p in graph.patches:
   graph.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
graph.set(title='Fraud vs No Fraud')
plt.show()

print(data.head())



data['dist_card_holder_to_merch'] = data.apply(lambda row : 
    hs.haversine((row['lat'],row['long']),(row['merch_lat'],row['merch_long'])), axis=1)

print(data[['lat','long','merch_lat','merch_long','dist_card_holder_to_merch']].head())

data['gender']=np.where(data['gender'] =='F', 0, data['gender'])
data['gender']=np.where(data['gender'] =='M', 1, data['gender'])


print(data.head())


cat_vars=['category']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['category']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values

print(data_final.columns.values)

data = data_final[['category_entertainment',
'category_food_dining','category_gas_transport','category_grocery_net',
'category_grocery_pos','category_health_fitness','category_home',
'category_kids_pets','category_misc_net','category_misc_pos',
'category_personal_care','category_shopping_net','category_shopping_pos',
'category_travel','amt','gender','dist_card_holder_to_merch','is_fraud']]
print("oi")
print(data.head())

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#charts - exploratory analysis

# #category_entertainment vs is_fraud
# table=pd.crosstab(data.category_entertainment,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_entertainment vs Fraud')
# plt.xlabel('category_entertainment')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_food_dining vs is_fraud
# table=pd.crosstab(data.category_food_dining,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_food_dining vs Fraud')
# plt.xlabel('category_food_dining')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_gas_transport vs is_fraud
# table=pd.crosstab(data.category_gas_transport,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_gas_transport vs Fraud')
# plt.xlabel('category_gas_transport')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_grocery_net vs is_fraud
# table=pd.crosstab(data.category_grocery_net,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_grocery_net vs Fraud')
# plt.xlabel('category_grocery_net')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_health_fitness vs is_fraud
# table=pd.crosstab(data.category_health_fitness,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_health_fitness vs Fraud')
# plt.xlabel('category_health_fitness')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_home vs is_fraud
# table=pd.crosstab(data.category_home,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_home vs Fraud')
# plt.xlabel('category_home')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_kids_pets vs is_fraud
# table=pd.crosstab(data.category_kids_pets,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_kids_pets vs Fraud')
# plt.xlabel('category_kids_pets')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_misc_net vs is_fraud
# table=pd.crosstab(data.category_misc_net,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_misc_net vs Fraud')
# plt.xlabel('category_misc_net')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_misc_pos vs is_fraud
# table=pd.crosstab(data.category_misc_pos,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_misc_pos vs Fraud')
# plt.xlabel('category_misc_pos')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_personal_care vs is_fraud
# table=pd.crosstab(data.category_personal_care,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_personal_care vs Fraud')
# plt.xlabel('category_personal_care')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_shopping_net vs is_fraud
# table=pd.crosstab(data.category_shopping_net,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_shopping_net vs Fraud')
# plt.xlabel('category_shopping_net')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_shopping_pos vs is_fraud
# table=pd.crosstab(data.category_shopping_pos,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_shopping_pos vs Fraud')
# plt.xlabel('category_shopping_pos')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #category_travel vs is_fraud
# table=pd.crosstab(data.category_travel,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_travel vs Fraud')
# plt.xlabel('category_travel')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #amt vs is_fraud
# sns.boxplot(data=data  , x="is_fraud", y="amt")
# plt.show()

# #amt vs is_fraud with filter
# Q1 = data['amt'].quantile(0.25)
# Q3 = data['amt'].quantile(0.75)
# IQR = Q3 - Q1    #IQR is interquartile range. 
# filter = (data['amt'] >= Q1 - 1.5 * IQR) & (data['amt'] <= Q3 + 1.5 *IQR)
# data.loc[filter]  
# sns.boxplot(data=data.loc[filter]  , x="is_fraud", y="amt")
# plt.show()

# #gender vs is_fraud
# table=pd.crosstab(data.gender,data.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of gender vs Fraud')
# plt.xlabel('gender')
# plt.ylabel('Proportion of Fraud')
# plt.show()

# #dist_card_holder_to_merch vs is_fraud
# sns.boxplot(data=data  , x="is_fraud", y="dist_card_holder_to_merch")
# plt.show()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


X_original = data.loc[:, data.columns != 'is_fraud']
y_original = data.loc[:, data.columns == 'is_fraud']

y_original = y_original.astype('int')




from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)
X_train, X_test_original_train, y_train, y_test_original_train = train_test_split(X_original, y_original, test_size=0.3, random_state=0)
columns = X_train.columns
print(type(y_train))
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['is_fraud'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no fraud in oversampled data",len(os_data_y[os_data_y['is_fraud']==0]))
print("Number of fraud",len(os_data_y[os_data_y['is_fraud']==1]))
print("Proportion of no fraud data in oversampled data is ",len(os_data_y[os_data_y['is_fraud']==0])/len(os_data_X))
print("Proportion of fraud data in oversampled data is ",len(os_data_y[os_data_y['is_fraud']==1])/len(os_data_X))


data_oversampled=pd.concat([os_data_X, os_data_y], axis=1)
print(data_oversampled.head())


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#charts - exploratory analysis oversampled


# #gender vs is_fraud
# table=pd.crosstab(data_oversampled.gender,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of gender vs Fraud')
# plt.xlabel('gender')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_entertainment vs is_fraud
# table=pd.crosstab(data_oversampled.category_entertainment,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_entertainment vs Fraud')
# plt.xlabel('category_entertainment')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_food_dining vs is_fraud
# table=pd.crosstab(data_oversampled.category_food_dining,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_food_dining vs Fraud')
# plt.xlabel('category_food_dining')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_gas_transport vs is_fraud
# table=pd.crosstab(data_oversampled.category_gas_transport,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_gas_transport vs Fraud')
# plt.xlabel('category_gas_transport')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_grocery_net vs is_fraud
# table=pd.crosstab(data_oversampled.category_grocery_net,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_grocery_net vs Fraud')
# plt.xlabel('category_grocery_net')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_health_fitness vs is_fraud
# table=pd.crosstab(data_oversampled.category_health_fitness,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_health_fitness vs Fraud')
# plt.xlabel('category_health_fitness')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_home vs is_fraud
# table=pd.crosstab(data_oversampled.category_home,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_home vs Fraud')
# plt.xlabel('category_home')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_kids_pets vs is_fraud
# table=pd.crosstab(data_oversampled.category_kids_pets,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_kids_pets vs Fraud')
# plt.xlabel('category_kids_pets')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_misc_net vs is_fraud
# table=pd.crosstab(data_oversampled.category_misc_net,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_misc_net vs Fraud')
# plt.xlabel('category_misc_net')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_misc_pos vs is_fraud
# table=pd.crosstab(data_oversampled.category_misc_pos,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_misc_pos vs Fraud')
# plt.xlabel('category_misc_pos')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_personal_care vs is_fraud
# table=pd.crosstab(data_oversampled.category_personal_care,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_personal_care vs Fraud')
# plt.xlabel('category_personal_care')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_shopping_net vs is_fraud
# table=pd.crosstab(data_oversampled.category_shopping_net,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_shopping_net vs Fraud')
# plt.xlabel('category_shopping_net')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_shopping_pos vs is_fraud
# table=pd.crosstab(data_oversampled.category_shopping_pos,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_shopping_pos vs Fraud')
# plt.xlabel('category_shopping_pos')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #category_travel vs is_fraud
# table=pd.crosstab(data_oversampled.category_travel,data_oversampled.is_fraud)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of category_travel vs Fraud')
# plt.xlabel('category_travel')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #amt vs is_fraud
# sns.boxplot(data=data_oversampled  , x="is_fraud", y="amt")
# plt.show()

# #amt vs is_fraud with filter
# Q1 = data_oversampled['amt'].quantile(0.25)
# Q3 = data_oversampled['amt'].quantile(0.75)
# IQR = Q3 - Q1    #IQR is interquartile range. 
# filter = (data_oversampled['amt'] >= Q1 - 1.5 * IQR) & (data_oversampled['amt'] <= Q3 + 1.5 *IQR)
# data_oversampled.loc[filter]  
# sns.boxplot(data=data_oversampled.loc[filter]  , x="is_fraud", y="amt")
# plt.show()

# print(data_oversampled.columns)
# print(data_oversampled.head())

# #gender vs is_fraud
# table=pd.crosstab(data_oversampled['gender'].astype('int'),data_oversampled['is_fraud'].astype('int'))
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
# plt.title('Stacked Bar Chart of gender vs Fraud')
# plt.xlabel('gender')
# plt.ylabel('Proportion of Fraud')

# plt.show()

# #dist_card_holder_to_merch vs is_fraud
# sns.boxplot(data=data_oversampled  , x="is_fraud", y="dist_card_holder_to_merch")
# plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++











data_final_vars=data.columns.values.tolist()
y=['is_fraud']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=2000)
rfe = RFE(logreg, n_features_to_select=1)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

array_df=[]
array_df.extend(data.columns.values)
array_df.remove('is_fraud')
print(data.columns.values)



X=os_data_X[array_df]

print(X.head())




cols=['category_entertainment', 'category_food_dining', 'category_gas_transport',
 'category_grocery_net', 'category_grocery_pos', 'category_health_fitness',
 'category_home', 'category_kids_pets', 'category_misc_net',
 'category_misc_pos', 'category_personal_care', 'category_shopping_net',
 'category_shopping_pos', 'category_travel', 'amt', 'gender',
 'dist_card_holder_to_merch']

#cols=['category_grocery_net', 'category_health_fitness',
#'category_home', 'category_kids_pets', 'category_travel','category_personal_care','category_misc_pos','category_food_dining','category_entertainment']

X=os_data_X[cols]
y=os_data_y['is_fraud']




from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

logit_roc_auc_train = roc_auc_score(y_train, logreg.predict(X_train))
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, logreg.predict_proba(X_train)[:,1])
plt.figure()
plt.plot(fpr_train, tpr_train, label='Logistic Regression Train - Train Dataset(area = %0.5f)' % logit_roc_auc_train)


#plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression Test - Train Dataset(area = %0.5f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

r_squared = logreg.score(X, y)
print(r_squared)




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# with the Test Set

#read and treat data
data = pd.read_csv(r'C:\Users\baral\OneDrive\Documents\Engenharia de Dados - USP\Análises Preditivas\Trabalho\dataset\fraudTest.csv')

data = data.dropna()





data['dist_card_holder_to_merch'] = data.apply(lambda row : 
    hs.haversine((row['lat'],row['long']),(row['merch_lat'],row['merch_long'])), axis=1)

print(data[['lat','long','merch_lat','merch_long','dist_card_holder_to_merch']].head())

data['gender']=np.where(data['gender'] =='F', 0, data['gender'])
data['gender']=np.where(data['gender'] =='M', 1, data['gender'])

#data.astype({'is_fraud': 'int32'}).dtypes

print(data.head())


cat_vars=['category']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['category']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values

print(data_final.columns.values)

data = data_final[['category_entertainment',
'category_food_dining','category_gas_transport','category_grocery_net',
'category_grocery_pos','category_health_fitness','category_home',
'category_kids_pets','category_misc_net','category_misc_pos',
'category_personal_care','category_shopping_net','category_shopping_pos',
'category_travel','amt','gender','dist_card_holder_to_merch','is_fraud']]

print(data.head())






X_original_test_set = data.loc[:, data.columns != 'is_fraud']
y_original_test_set = data.loc[:, data.columns == 'is_fraud']


















# example of random undersampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
# define dataset
# X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# summarize class distribution
print(Counter(y_original_test_set))
# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_over, y_over = undersample.fit_resample(X_original_test_set, y_original_test_set)
# summarize class distribution
print(Counter(y_over))










y_pred_test_set = logreg.predict(X_over)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_over, y_over)))
from sklearn.metrics import accuracy_score
print(roc_auc_score(y_over,y_pred_test_set))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_over, y_pred_test_set)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_over, y_pred_test_set))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_over, logreg.predict(X_over))
fpr, tpr, thresholds = roc_curve(y_over, logreg.predict_proba(X_over)[:,1])

logit_roc_auc_train = roc_auc_score(y_train, logreg.predict(X_train))
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, logreg.predict_proba(X_train)[:,1])
plt.figure()
plt.plot(fpr_train, tpr_train, label='Logistic Regression Train - Test Dataset (area = %0.5f)' % logit_roc_auc_train)


#plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression - Test Dataset (area = %0.5f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()