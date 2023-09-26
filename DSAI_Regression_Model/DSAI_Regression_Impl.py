"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
© Copyright 2022, California, Department of Motor Vehicle, all rights reserved.
The source code and all its associated artifacts belong to the California Department of Motor Vehicle (CA, DMV), and no one has any ownership
and control over this source code and its belongings. Any attempt to copy the source code or repurpose the source code and lead to criminal
prosecution. Don't hesitate to contact DMV for further information on this copyright statement.

Release Notes and Development Platform:
The source code was developed on the Google Cloud platform using Google Cloud Functions serverless computing architecture. The Cloud
Functions gen 2 version automatically deploys the cloud function on Google Cloud Run as a service under the same name as the Cloud
Functions. The initial version of this code was created to quickly demonstrate the role of MLOps in the ELP process and to create an MVP. Later,
this code will be optimized, and Python OOP concepts will be introduced to increase the code reusability and efficiency.
____________________________________________________________________________________________________________
Development Platform                | Developer       | Reviewer   | Release  | Version  | Date
____________________________________|_________________|____________|__________|__________|__________________
Google Cloud Serverless Computing   | DMV Consultant  | Ajay Gupta | Initial  | 1.0      | 09/18/2022

-----------------------------------------------------------------------------------------------------------------------------------------------------
"""

import streamlit as vAR_st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from lime import lime_tabular
import streamlit.components.v1 as components
import base64




def Regression_Model():
    
    # if "vAR_model" not in vAR_st.session_state: 
    #     vAR_st.session_state = {}
    
    # try:

    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_result_data = pd.DataFrame()
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Upload Training Data')

    with col4:
        vAR_st.write('')
        vAR_dataset = vAR_st.file_uploader("Choose a file",type=['csv'],key="reg")
        
    if vAR_dataset is not None:
        vAR_outcome_analysis = False
        vAR_test_data = None
        vAR_test = False
        vAR_tested = False
        
        vAR_df = pd.read_csv(vAR_dataset)
        
        Preview_Data(vAR_df,"Training")
            
        Statistics_Details(vAR_df)
        
        Feature_Selection(vAR_df)
        
        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
        
        
            
        with col2:
            
            vAR_st.write('')
            vAR_st.write('')
            
            vAR_st.subheader('Model Training')
            
        with col4:
            
            vAR_st.write('')
            vAR_st.write('')
        
            vAR_model_train = vAR_st.button("Train the Model")
            vAR_st.write('')
            
            vAR_st.write('')
            
                    
        if vAR_model_train:
            
            if "vAR_model_reg" not in vAR_st.session_state and "X_train_reg" not in vAR_st.session_state:
                vAR_st.session_state['vAR_model_reg'],vAR_st.session_state['X_train_reg'] = Train_Model(vAR_df)
            
        
        if vAR_st.session_state['vAR_model_reg'] is not None:
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            with col2:
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.subheader('Upload Test Data')

            with col4:
                vAR_st.write('')
                vAR_test_dataset = vAR_st.file_uploader("Choose a file",type=['csv'],key="test")
                
        if vAR_test_dataset is not None:
                
            vAR_test_data = pd.read_csv(vAR_test_dataset)
        
            
            # Preview Test Data
            vAR_test_data.drop('Risk_Score', axis=1,errors="ignore",inplace=True)
            Preview_Data(vAR_test_data,"Test")
            
            
            
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            with col2:
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.subheader('Test Model')

            with col4:
                vAR_st.write('')
                vAR_st.write('')
                vAR_test = vAR_st.button("Test Model")
                
                
                
        if vAR_test:
            
            vAR_model = vAR_st.session_state['vAR_model_reg']
            print('model - ',vAR_model)
            
            col1,col2,col3 = vAR_st.columns([3,15,1])
            
            with col2:
                vAR_df_columns = vAR_test_data.columns
        
                vAR_numeric_columns = vAR_test_data._get_numeric_data().columns 
                
                vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
                
                data_encoded = pd.get_dummies(vAR_test_data, columns=vAR_categorical_column)
                
                vAR_test_data["Predicted_Score"] = vAR_model.predict(data_encoded)
                
                if "vAR_test_data_reg" not in vAR_st.session_state:
                    vAR_st.session_state["vAR_test_data_reg"] = vAR_test_data
                
                vAR_st.write('')
                vAR_st.write('')
                vAR_st.write(vAR_test_data)
                if not vAR_tested:
                    vAR_st.session_state["vAR_tested"] = True
                
        print('vaR_tested - ',vAR_st.session_state["vAR_tested"])
        if vAR_st.session_state["vAR_tested"]:
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            
            with col4:
                
                vAR_st.markdown(create_download_button(vAR_test_data), unsafe_allow_html=True)
                
#                 vAR_result_data = vAR_test_data.to_csv().encode('utf-8')
                
#                 vAR_st.download_button(
# label="Download Model Outcome as CSV",
# data=vAR_result_data,
# file_name='Model Outcome.csv',
# mime='text/csv',
# )
                
            vAR_st.write('')
            vAR_st.write('')
        
        if vAR_st.session_state["vAR_tested"]:
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
                
            with col4:
                vAR_outcome_analysis = vAR_st.button("Model Outcome Analysis")
                
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            if vAR_outcome_analysis:
                
                vAR_df_columns = vAR_test_data.columns
        
                vAR_numeric_columns = vAR_test_data._get_numeric_data().columns 
                
                vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
                
                df = vAR_st.session_state["vAR_test_data_reg"].drop(vAR_categorical_column,axis=1)
                
                
                
                with col2:
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.info("Correlation Between Features")
                    plot_correlation_matrix(df)
                        
                    
                with col4:
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.info("Distribution of Predicted_Score")
                    plot_score_distribution(df)
                    
                    
                    
                col1,col2,col3 = vAR_st.columns([1,15,1])
                
                with col2:
                    vAR_st.markdown('<hr style="border:2px solid gray;">', unsafe_allow_html=True)
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.markdown("<div style='text-align: center; color: black;'>Risk Score Variation with Features in ScatterPlot</div>", unsafe_allow_html=True)
                    vAR_st.write('')
                    vAR_st.write('')
                    plot_scatter_matrix(vAR_st.session_state["vAR_test_data_reg"])
                    
            col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            
            with col2:
                vAR_st.write('')
                vAR_st.subheader("Select Driver ID For XAI")
                
                
                
            with col4:
                
                vAR_idx_values = ['Select Driver Id'] 
                vAR_idx_values.extend(["DL10"+str(item) for item in vAR_test_data.index])               
                vAR_test_id = vAR_st.selectbox(' ',vAR_idx_values)
                
                
            
            if vAR_test_id!='Select Driver Id': 
                vAR_df_columns = vAR_test_data.columns
        
                vAR_numeric_columns = vAR_test_data._get_numeric_data().columns 
                
                vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
                data_encoded = pd.get_dummies(vAR_test_data, columns=vAR_categorical_column)
                
                # data_encoded.rename(columns = {'Vehicle_Type_Motorcycle':'Vehicle_Type'}, inplace = True)
                # data_encoded2 = data_encoded.drop(["Predicted_Score"],axis=1,errors='ignore')
                features = data_encoded.columns
                col1,col2,col3 = vAR_st.columns([1,15,1])
                
                with col2:
                    vAR_st.write('')
                    vAR_st.markdown('<hr style="border:2px solid gray;">', unsafe_allow_html=True)
                    
                    vAR_st.write('')
                    vAR_st.markdown("<div style='text-align: center; color: black;font-weight:bold;'>Explainable AI with LIME(Local  Interpretable Model-agnostic Explanations) Technique</div>", unsafe_allow_html=True)
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.write('')
                    vAR_st.write('')
                    ExplainableAI(vAR_st.session_state["X_train_reg"],features,vAR_st.session_state["vAR_model_reg"],data_encoded,vAR_test_id)
                    
                
                        
    # except BaseException as e:
    #     print('Exception occurs in regression model - ',str(e))
        
    #     print('Exception traceback occurs in regression model - ',traceback.print_exc(str(e)))
                    
                    
                    
                    
                
            
            
        
    
        
        
        
                    
            
                
                
            
def Preview_Data(vAR_df,type):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Preview '+type +' Data')
        

    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_preview_data = vAR_st.button("Preview "+type+" Data")
        
        
    if vAR_preview_data:
        
        if type=="Test":
            col1,col2,col3 = vAR_st.columns([3,10,1.5])
        else:
            col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
        
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.dataframe(data=vAR_df,height=310)
            # vAR_st.info(" Note: Risk Score Calculated Based on Feature Weightage")
            
        
            
        
            


def Statistics_Details(vAR_df):
    
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
            
    with col2:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Stats on Training Data')
        
    with col4:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_data_stats = vAR_st.button("View Training Data Stats")
                
    if vAR_data_stats:
        
        col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
        vAR_st.write('')
        vAR_st.write('')
        
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write(vAR_df.describe(),height=200)

def Feature_Selection(vAR_df):
    
    vAR_columns =["All"]
    col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
    vAR_columns.extend(list(vAR_df.columns))
    
    if "Risk_Score" in vAR_columns:
        vAR_columns.remove("Risk_Score")
    # To remove last column(target column)
    vAR_columns.pop()
    if "Normalized_Risk_Score" in vAR_columns:
        vAR_columns.remove("Normalized_Risk_Score")
    
    
            
    with col2:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader('Feature Selection')
        
    with col4:
        
        vAR_st.write('')
        vAR_st.write('')
        vAR_features = vAR_st.multiselect(' ',vAR_columns,default="All")
        vAR_st.write('')
        vAR_st.write('')
        
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    with col2:
        vAR_st.write('')
        with vAR_st.expander("List selected features"):  
            if 'All' in vAR_features:
                vAR_st.write('Features:',vAR_columns[1:])
            else:
                for i in range(0,len(vAR_features)):
                    vAR_st.write('Feature',i+1,':',vAR_features[i])
        

    
    

def Train_Model(vAR_df):
    col1,col2,col3 = vAR_st.columns([1.5,10,1.5])
    
    with col2:
        # Preprocessing
        
        vAR_df_columns = vAR_df.columns
        
        vAR_numeric_columns = vAR_df._get_numeric_data().columns 
        
        vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
        
        vAR_train_df = vAR_df.drop(vAR_df.columns[-1],axis=1)
        
        
        data_encoded = pd.get_dummies(vAR_train_df, columns=vAR_categorical_column)
        
        

        # Define features and target
        # Remove this later, for dataset which doesn't contain Risk_Score column
        X = data_encoded.drop('Risk_Score', axis=1,errors="ignore",inplace=True)
        
        X = data_encoded
        
        y = vAR_df.iloc[: , -1:]
        
        print('xcols - ',X.columns)
        print('ycols - ',y.columns)

        # Splitting the Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=42)
        
        
        # print('X_test - ',X_test.iloc[0])
        vAR_st.info("Data Preprocessing Completed!")
        vAR_st.info("Regression Model Successfully Trained")
        


        # Training the Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        
        # y_pred = model.predict([[ 7,  9, 15, 56, 80,  1]])
        
        # vAR_st.write(y_pred)

        
    return model,X_train


# Model Outcome Analysis Functions
    
def plot_correlation_matrix(data):
    print('df len - ',len(data))
    correlation_matrix = data.corr()
    plt.figure(figsize=(8, 8))
    # plt.title("Correlation Between Features")
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    vAR_st.pyplot(plt)
    
    
def plot_score_distribution(data,x='Predicted_Score'):
    sns.displot(data['Predicted_Score'])
    # plt.title(f'Distribution of Predicted_Score')
    vAR_st.pyplot(plt)


def plot_scatter_matrix(df):
    
    vAR_df_columns = df.columns
        
    vAR_numeric_columns = df._get_numeric_data().columns 
    
    vAR_categorical_column = list(set(vAR_df_columns) - set(vAR_numeric_columns))
    
    if len(vAR_categorical_column)>0:
        sns.pairplot(df,hue=vAR_categorical_column[0],kind='scatter')
    else:
        sns.pairplot(df,kind='scatter')
    vAR_st.pyplot(plt)
    
def ExplainableAI(X_train,features,vAR_model,vAR_test_data,test_id):
    
    print('train cols - ',(X_train.columns))
    print('feature cols - ',(features))
    
    explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=features,
                                                   verbose=True,
                                                   mode='regression')
    
    exp = explainer_lime.explain_instance(
    vAR_test_data.iloc[int(test_id[4:])], vAR_model.predict)
    
    # Extract LIME explanations and visualize
    features, weights = zip(*exp.as_list())
    chart_data = pd.DataFrame({'Feature': features, 'Weight': weights})
    print(chart_data.head())
    
    
    
    vAR_st.dataframe(chart_data)
    
    vAR_st.write('')
    
    html = exp.as_html()
    components.html(html, height=1000,width=1000)
    
    

def dataframe_to_base64(df):
    """Convert dataframe to base64-encoded csv string for downloading."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return b64

def create_download_button(df, filename="data.csv"):
    """Generate a link to download the dataframe as a csv file."""
    b64_csv = dataframe_to_base64(df)
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="{filename}" style="display: block; margin: 1em 0; padding: 13px 20px 16px 12px; background-color: rgb(47 236 106); text-align: center; border: none; border-radius: 6px;color: black; text-decoration: none;">Download Model Outcome as CSV</a>'
    return href



# Logic for Risk Score Calculation

# weights = {
#     'Conviction Points': 10,
#     'Geo Risk': 5,
#     'Years of Experience': -3,
#     'Vehicle Type': {'Car': 0, 'Motorcycle': 10},
#     'Age': lambda age: 10 if age < 25 else 0
# }

# # Calculate risk score for each row
# data['Risk Score'] = (
#     data['Conviction Points'] * weights['Conviction Points'] +
#     data['Geo Risk'] * weights['Geo Risk'] +
#     data['Years of Experience'] * weights['Years of Experience'] +
#     data['Vehicle Type'].map(weights['Vehicle Type']) +
#     data['Age'].apply(weights['Age'])
# )