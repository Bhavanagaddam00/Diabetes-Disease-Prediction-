import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report 
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency

""" PART 2
Importing Data """
df = pd.read_csv('diabetes.csv')
#print ("Data Head : \n", df.head())
#print ("\nData Tail : \n", df.tail())
#print("\nSample : \n", df.sample(5))
#print("\nData Types:\n",df.dtypes)#datatypes
#print("\nData Info:\n",df.info())
#print("\nData Shape:\n",df.shape)#dimensions
#print ("\n\nData Describe : \n", df.describe())#statistical summary
#summary=df.describe(include='all')
""" PART 3
Checking Null Values """
#for c in summary.columns:
#    print(c,np.sum(summary[c].isnull()))  
    
""" PART 4
Remove Outliers """
#for i in df:
#    if df[i].dtype=='int64' and i!='Outcome':
#        df[i]=df[i].replace(0,df[i].mean())
        
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
df=remove_outlier(df,'Insulin')
df=remove_outlier(df,'BMI')
df=remove_outlier(df,'DiabetesPedigreeFunction')
df=remove_outlier(df,'Age')
df=remove_outlier(df,'Pregnancies')
df=remove_outlier(df,'BloodPressure')
df=remove_outlier(df,'SkinThickness')
upper = df.corr().where(np.triu(np.ones(df.corr().shape),
                                      k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.97)]
#print(to_drop)
df = df.drop(df[to_drop], axis=1)
#print(df.columns)
#print(sns.boxplot(x=df["SkinThickness"]))
#plt.figure(figsize=(20,5))
#sns.heatmap(df.corr(),annot=True,cmap='YlGnBu', vmin=None, vmax=None,linewidths=0)
#df=df.drop(['Age'],axis=1)
#print(df)
#print(sns.boxplot(x=df["Insulin"]))
df['Age']=pd.cut(df['Age'],3,labels=['young','middle','old'])
df['Age'] = df['Age'].map({"young":0,"middle":1,"old":2})
#print(df)
df['Glucose']=pd.cut(df['Glucose'],3,labels=['low','medium','high'])
df['Glucose'] = df['Glucose'].map({"low":0,"medium":1,"high":2})
#print(df)
df['BloodPressure']=pd.cut(df['BloodPressure'],3,labels=['low','medium','high'])
df['BloodPressure'] = df['BloodPressure'].map({"low":0,"medium":1,"high":2})
#print(df)
df['Insulin']=pd.cut(df['Insulin'],3,labels=['low','medium','high'])
df['Insulin'] = df['Insulin'].map({"low":0,"medium":1,"high":2})
#print(df)
df['BMI']=pd.cut(df['BMI'],3,labels=['low','medium','high'])
df['BMI'] = df['BMI'].map({"low":0,"medium":1,"high":2})
#print(df)
df['DiabetesPedigreeFunction']=pd.cut(df['DiabetesPedigreeFunction'],3,labels=['normal','prediabetes','diabetes'])
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].map({"normal":0,"prediabetes":1,"diabetes":2})
#print(df)
y=df.Outcome
x=df.drop('Outcome',axis=1)

'''
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=100)
##standard scalar
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

l=[]
l2=[]
#print(df.dtypes)
for i in df:
    if df[i].dtype=='float64':
        l2.append(i)
    else:
        l.append(i)
#print(l)
#print(l2)
def chi_sqaure(i,j):
    table=pd.crosstab(i,j) 
    stat,p,dof,expected=chi2_contingency(table)
#    print(p)
    alpha=0.05
    if p<=alpha:
        return True
    else:
        return False
for i in l:
    d=chi_sqaure(df['Outcome'],df[i])
    if d==False:
        df=df.drop(i,axis=1)
#print(df.dtypes)
        
##PCA
pca = PCA()   
x_train = pca.fit_transform(x_train) 
x_test = pca.transform(x_test) 
explained_variance = pca.explained_variance_ratio_
#print(x_train)
#print(x_test)

""" PART 5
Applying Algorithms """
acc=[]

##Smote test
sm = SMOTE(random_state = 2) 
x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())


lr1 = LogisticRegression() 
#lr1.fit(x_train, y_train) 
lr1.fit(x_train_res, y_train_res) 
predictions = lr1.predict(x_test) 
# print classification report 
#print(classification_report(y_test, predictions)) 
print("Accuracy percentage LR:"+"{:.2f}".format(accuracy_score(y_test,predictions)*100))
lr_accuracy_score=accuracy_score(y_test,predictions)*100
acc.append(round(lr_accuracy_score,2))
#confusion matrix
cm = confusion_matrix(y_test, predictions)
#print(cm)
# Saving model to disk
pickle.dump(lr1, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))


##SupportVectorClassifier
S=SVC()
#Train the model
S.fit(x_train_res,y_train_res)
#predicting outcome
y_pre_S=S.predict(x_test)
print("Accuracy percentage SVC:"+"{:.2f}".format(accuracy_score(y_test,y_pre_S)*100))
svc_accuracy_score=accuracy_score(y_test,y_pre_S)*100
acc.append(round(svc_accuracy_score,2))
#confusion matrix
cm = confusion_matrix(y_test, y_pre_S)
#print(cm)
# Saving model to disk
pickle.dump(S, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))


##NaiveBayesClassifier
gnb = GaussianNB()
#Train the model
gnb.fit(x_train_res,y_train_res)
#predicting outcome
y_pre_gnb=gnb.predict(x_test)
print("Accuracy percentage GNB:"+"{:.2f}".format(accuracy_score(y_test,y_pre_gnb)*100))
gnb_accuracy_score=accuracy_score(y_test,y_pre_gnb)*100
acc.append(round(gnb_accuracy_score,2))
#confusion matrix
cm = confusion_matrix(y_test, y_pre_gnb)
#print(cm)
# Saving model to disk
pickle.dump(gnb, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))


##DecisionTreeClassifier
DTC=DecisionTreeClassifier()
#Train the model
DTC.fit(x_train_res,y_train_res)
#predicting outcome
y_pre_DTC=DTC.predict(x_test)
print("Accuracy percentage DTC:"+"{:.2f}".format(accuracy_score(y_test,y_pre_DTC)*100))
dtc_accuracy_score=accuracy_score(y_test,y_pre_DTC)*100
acc.append(round(dtc_accuracy_score,2))
#confusion matrix
cm = confusion_matrix(y_test, y_pre_DTC)
#print(cm)
# Saving model to disk
pickle.dump(DTC, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))


##RandomForestClassifier
RF=RandomForestClassifier()
#Train the model
RF.fit(x_train_res,y_train_res)
#predicting outcome
y_pre_RF=RF.predict(x_test)
print("Accuracy percentage RF:"+"{:.2f}".format(accuracy_score(y_test,y_pre_RF)*100))
rf_accuracy_score=accuracy_score(y_test,y_pre_RF)*100
acc.append(round(rf_accuracy_score,2))
#confusion matrix
cm = confusion_matrix(y_test, y_pre_RF)
#print(cm)
# Saving model to disk
pickle.dump(RF, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))


##KNeighborsClassifier
KNC=KNeighborsClassifier()
#Train the model
KNC.fit(x_train_res,y_train_res)
#predicting outcome
y_pre_KNC=KNC.predict(x_test)
print("Accuracy percentage KNC:"+"{:.2f}".format(accuracy_score(y_test,y_pre_KNC)*100))
knc_accuracy_score=accuracy_score(y_test,y_pre_KNC)*100
acc.append(round(knc_accuracy_score,2))
#confusion matrix
cm = confusion_matrix(y_test, y_pre_KNC)
#print(cm)
# Saving model to disk
pickle.dump(KNC, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))


models=['LR','SVC','DTS','RF','KNN','GNB']
fig = plt.figure()
import matplotlib
matplotlib.style.use('ggplot')
ax = fig.add_axes([0,0,1,1])
plt.title("Accuracy Score of different models ")
ax.bar(models,acc,width=0.8,color=['yellow','pink','skyblue','brown','green','orange'])

'''
from sklearn. model_selection import StratifiedKFold
accuracy=[]
skf = StratifiedKFold(n_splits=10, random_state=None)
skf.get_n_splits(x, y)

# X is the feature set and y is the target
for train_index, test_index in skf.split(x, y) :
    #print ("Train:", train_index, "validation:", test_index)
    X1_train, X1_test = x.iloc[train_index], x.iloc[test_index]
    y1_train, y1_test = y.iloc[train_index], y.iloc[test_index]
    ##standard scalar
    st_x=StandardScaler()
    X1_train=st_x.fit_transform(X1_train)
    X1_test=st_x.transform(X1_test)
    ##PCA
    pca = PCA()   
    X1_train = pca.fit_transform(X1_train) 
    X1_test = pca.transform(X1_test) 
    explained_variance = pca.explained_variance_ratio_
    ##Smote test
    sm = SMOTE(random_state = 2) 
    X1_train, y1_train = sm.fit_resample(X1_train, y1_train.ravel())
    ##chi-square test
    l=[]
    l2=[]
    #print(df.dtypes)
    for i in df:
        if df[i].dtype=='float64':
            l2.append(i)
        else:
            l.append(i)
    #print(l)
    #print(l2)
    def chi_sqaure(i,j):
        table=pd.crosstab(i,j) 
        stat,p,dof,expected=chi2_contingency(table)
    #    print(p)
        alpha=0.05
        if p<=alpha:
            return True
        else:
            return False
    for i in l:
        d=chi_sqaure(df['Outcome'],df[i])
        if d==False:
            df=df.drop(i,axis=1)
    #print(df.dtypes)
    classifier=LogisticRegression()
#    classifier=SVC()
#    classifier=GaussianNB()
#    classifier=DecisionTreeClassifier()
#    classifier=RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
#    classifier=KNeighborsClassifier()
    classifier.fit(X1_train, y1_train)
    prediction=classifier . predict(X1_test)
    score=accuracy_score(prediction, y1_test)
    accuracy.append(score)
    # Saving model to disk
    pickle.dump(classifier, open('model.pkl','wb'))
    # Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
    #print(model.predict([[2, 9, 6]]))
#print ((np.array(accuracy).mean())*100)
