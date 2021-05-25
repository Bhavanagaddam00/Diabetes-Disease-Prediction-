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
from sklearn.model_selection import StratifiedKFold

""" PART 2
Importing Data """
df = pd.read_csv('diabetes.csv')

""" PART 3
Checking Null Values """
#for c in summary.columns:
#    print(c,np.sum(summary[c].isnull()))  
    
""" PART 4
Remove Outliers """
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
df = df.drop(df[to_drop], axis=1)
df['Age']=pd.cut(df['Age'],3,labels=['young','middle','old'])
df['Age'] = df['Age'].map({"young":0,"middle":1,"old":2})
df['Glucose']=pd.cut(df['Glucose'],3,labels=['low','medium','high'])
df['Glucose'] = df['Glucose'].map({"low":0,"medium":1,"high":2})
df['BloodPressure']=pd.cut(df['BloodPressure'],3,labels=['low','medium','high'])
df['BloodPressure'] = df['BloodPressure'].map({"low":0,"medium":1,"high":2})
df['Insulin']=pd.cut(df['Insulin'],3,labels=['low','medium','high'])
df['Insulin'] = df['Insulin'].map({"low":0,"medium":1,"high":2})
df['BMI']=pd.cut(df['BMI'],3,labels=['low','medium','high'])
df['BMI'] = df['BMI'].map({"low":0,"medium":1,"high":2})
df['DiabetesPedigreeFunction']=pd.cut(df['DiabetesPedigreeFunction'],3,labels=['normal','prediabetes','diabetes'])
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].map({"normal":0,"prediabetes":1,"diabetes":2})
y=df.Outcome
x=df.drop('Outcome',axis=1)

accuracy=[]
skf = StratifiedKFold(n_splits=10, random_state=None)
skf.get_n_splits(x, y)

# x is the feature set and y is the target
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
    def chi_sqaure(i,j):
        table=pd.crosstab(i,j) 
        stat,p,dof,expected=chi2_contingency(table)
        alpha=0.05
        if p<=alpha:
            return True
        else:
            return False
    for i in l:
        d=chi_sqaure(df['Outcome'],df[i])
        if d==False:
            df=df.drop(i,axis=1)
#    classifier=LogisticRegression()
    classifier=SVC()
#    classifier=GaussianNB()
#    classifier=DecisionTreeClassifier()
#    classifier=RandomForestClassifier()
#    classifier=KNeighborsClassifier()
    classifier.fit(X1_train, y1_train)
    prediction=classifier . predict(X1_test)
    score=accuracy_score(prediction, y1_test)
    accuracy.append(score)
    # Saving model to disk
    pickle.dump(classifier, open('model.pkl','wb'))
    # Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
#print((np.array(accuracy).mean())*100)
