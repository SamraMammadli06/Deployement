import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score


icon = Image.open('images/icon.png')
logo = Image.open('images/logo.png')
banner = Image.open('images/banner.png')

st.set_page_config(layout = "wide",
                   page_title = "Python Deployement",
                   page_icon = icon)

st.title("Python Week8 - Deployement")
st.text("Machine Leaning Web Application with streamlit")
st.sidebar.image(logo)


page  = st.sidebar.selectbox("",["Home","EDA","Modelling"])


if page == 'Home':
    st.header("Home")
    st.image(banner,use_column_width="always")
    data = st.selectbox(
    'Select the data',
    ('Water potability', 'Loan prediction'))
    if data == 'Water potability':
        df = pd.read_csv('water_potability.csv')
    else:
        df = pd.read_csv('loan_pred.csv')
    st.markdown('Data')
    st.dataframe(df)
    st.markdown('Dataframe length')
    st.write(len(df))
    st.markdown('Data description')
    st.dataframe(df.describe())
    st.markdown('Number of null values')
    st.dataframe(df.isnull().sum())
    st.markdown('Presence of duplicates')
    st.write(len(df)!=len(df.drop_duplicates()))
    st.markdown('Balance of values of the label')
    st.dataframe(pd.DataFrame(df.iloc[:,-1].value_counts()))


    #Selecting all numerical columns
    numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()
    filtered_numerical_columns = [col for col in numerical_columns if len(df[col].unique()) > 20]


    #Creating boxplots for all numerical columns
    st.markdown('Showing outliers for numerical columns')
    fig, axes = plt.subplots(nrows=len(filtered_numerical_columns), figsize=(10, 10))
    for i, column in enumerate(filtered_numerical_columns):
        sns.boxplot(x=df[column], ax=axes[i])
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

    st.markdown('Analyzing multicolinearity')
    plt.figure(figsize=(10,10))
    st.pyplot(sns.heatmap(df.corr(),annot=True).figure)
    plt.clf()
    st.markdown('\n')

elif page == 'EDA':
    def outlier_treatment(cols):
        Q1,Q3 = np.percentile(cols,[25,75])
        IQR = Q3-Q1
        upper_bound = Q3 + (1.5*IQR)
        lower_bound = Q1 - (1.5*IQR)
        return upper_bound,lower_bound
    def describeData(df):
        st.dataframe(df)
        st.subheader("Statistical Values")
        df.describe().T
        st.subheader("Balance")
        st.bar_chart(df.iloc[:,-1].value_counts())
        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ["Columns","Counts"]
        c1,c2,c3 = st.columns([2.5,1.5,2.5])
        c1.subheader("Null Variables")
        c1.dataframe(null_df)

        c2.subheader("Imputation")
        cat_method = c2.radio('Categorical',['Mode','Backfill','Ffill'])
        num_method = c2.radio("Numerical",["Mode","Median"])

        c2.subheader("Feature Engeneering")
        balance = c2.checkbox("Under Sampling")
        outlier = c2.checkbox("Clean Outlier")
        if c2.button("DATA PREPOCESSING"):
            cat_array = df.iloc[:,:-1].select_dtypes(include="object").columns
            num_array = df.iloc[:,:-1].select_dtypes(exclude="object").columns
            if cat_array.size>0:
                if cat_method=="Mode":
                    imp_cat = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
                    df[cat_array]=imp_cat.fit_transform(df[cat_array])
                elif cat_method=="Backfill":
                    df[cat_array].fillna(method = 'backfill',inplace=True)
                else:
                    df[cat_array].fillna(method = 'ffill',inplace=True)

            if num_array.size>0:
                if num_method=="Mode":
                    imp_num = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
                    
                else:
                     imp_num = SimpleImputer(missing_values=np.nan,strategy='median')
                df[num_array]=imp_num.fit_transform(df[num_array])
            df.dropna(axis=0,inplace=True)
        if balance:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler()
            X = df.iloc[:,:-1]
            Y = df.iloc[:,:-1]
            X,Y = rus.fit_resample(X,Y)
            df=pd.concat([X,Y],axis=1)
        if outlier:
            for col in num_array:
                lower,upper = outlier_treatment(df[col])
                df[col]=np.clip(df[col],a_min= lower,a_max=upper)
        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ["Columns","Counts"]
        st.subheader("Balance")
        st.bar_chart(df.iloc[:,-1].value_counts())
        c1.subheader("Null Variables")
        c1.dataframe(null_df)

        heatmap= px.imshow(df.corr())
        st.plotly_chart(heatmap)
        st.dataframe(df)

    st.header("Exploratory Data Analysis")
    data = st.selectbox(
    'Select the data that you want to analyze',
    ('Water potability', 'Loan prediction'))
    if data == 'Water potability':
        df = pd.read_csv('water_potability.csv')
        
    else:
        df = pd.read_csv('loan_pred.csv')
    describeData(df)
    if os.path.exists('prepared_data.csv'):
        os.remove('prepared_data.csv')
        df.to_csv('prepared_data.csv')
    else:
        df.to_csv('prepared_data.csv')
else:
    
    df = pd.read_csv('prepared_data.csv').iloc[:,1:]
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()
    filtered_numerical_columns = [col for col in numerical_columns if len(df[col].unique()) > 20]
    categorical_columns = [col for col in df.columns if col not in filtered_numerical_columns]

    if list(df.columns)[-1] in categorical_columns:
        categorical_columns.remove(list(df.columns)[-1])
    
    else:
        filtered_numerical_columns.remove(list(df.columns)[-1])

    if str(df[list(df.columns)[-1]].dtype) == 'object':
        y = pd.get_dummies(y,drop_first=True) 
        y = y.iloc[:,0]

    
    tts = st.text_input('Enter the size of test data.')
    random_state = st.text_input('Enter the random state.')

    column1,column2=st.columns(2)

    with column1:
        st.header('Categorical columns')
        cat_encoding = st.radio(
        'Choose the method of encoding',
        ('OneHotEncoder', 'OrdinalEncoder'))

    with column2:
        st.header('Numerical columns')
        num_scaling = st.radio(
        'Choose the method of scaling',
        ('MinMaxScaler', 'StandardScaler', 'RobustScaler'))

    model_choice = st.radio(
    'Choose the model',
    ('SVM', 'Logistic Regression', 'Random Forest'))

    if st.button('Build model')==True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(tts), random_state=int(random_state))


        if len(categorical_columns)>0:
            if cat_encoding == 'OneHotEncoder':
                encoder = OneHotEncoder(handle_unknown='ignore')
                X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
                X_test_encoded = encoder.transform(X_test[categorical_columns])
                X_train_encoded = X_train_encoded.toarray()
                X_test_encoded = X_test_encoded.toarray()
                X_train = pd.concat([X_train.drop(categorical_columns, axis=1).reset_index().drop('index',axis=1), pd.DataFrame(X_train_encoded)], axis=1)
                X_test = pd.concat([X_test.drop(categorical_columns, axis=1).reset_index().drop('index',axis=1), pd.DataFrame(X_test_encoded)], axis=1)
            else:
                encoder = OrdinalEncoder()
                X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])
                X_test[categorical_columns] = encoder.transform(X_test[categorical_columns])
                
        if len(filtered_numerical_columns) > 0:
            if num_scaling == 'MinMaxScaler':
                scaler = MinMaxScaler()
                X_train[filtered_numerical_columns] = scaler.fit_transform(X_train[filtered_numerical_columns])
                X_test[filtered_numerical_columns] = scaler.transform(X_test[filtered_numerical_columns])
            elif num_scaling == 'StandardScaler':
                scaler = StandardScaler()
                X_train[filtered_numerical_columns] = scaler.fit_transform(X_train[filtered_numerical_columns])
                X_test[filtered_numerical_columns] = scaler.transform(X_test[filtered_numerical_columns])
            else:
                scaler = RobustScaler()
                X_train[filtered_numerical_columns] = scaler.fit_transform(X_train[filtered_numerical_columns])
                X_test[filtered_numerical_columns] = scaler.transform(X_test[filtered_numerical_columns])
        

        if model_choice == 'SVM':
            model = svm.SVC(kernel='poly',probability=True,random_state=int(random_state))
            model.fit(X_train, y_train)
        elif model_choice == 'Logistic Regression':
            model = LogisticRegression(max_iter = 100000,random_state=int(random_state))
            model.fit(X_train, y_train)
        else:
            model = RandomForestClassifier(n_estimators = 100,random_state=int(random_state))
            model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        accuracy = round(accuracy_score(y_test, y_pred),2)

        st.write('The accuracy score of the model is: ', accuracy)

        st.write('Confusion matrix')

        st.write(confusion_matrix(y_test, y_pred))

        st.write('ROC AUC SCORE')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots(figsize=(10,10))
        logit_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        st.pyplot(fig)
        plt.clf()
        