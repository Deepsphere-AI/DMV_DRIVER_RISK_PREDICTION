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
import traceback
vAR_st.set_page_config(page_title="DMV Recommendation", layout="wide")

from DSAI_Utility.DSAI_Utility import All_Initialization,CSS_Property
from DSAI_VertexAI_Model.DSAI_Risk_Prediction import DriverRiskPrediction
from DSAI_VertexAI_Model.DSAI_Risk_Classification import DriverRiskClassification

from DSAI_Classification_Model.DSAI_Driver_Risk_Classification import DriverRiskClassification
from DSAI_Regression_Model.DSAI_Regression_Impl import Regression_Model
from DSAI_Classification_Model.DSAI_Classification_Impl import Classification_Model
from DSAI_RAG_LLM.DSAI_RAG_LLM_Impl import LLM_RAG

import os
import base64

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\ds_007\Downloads\elp-2022-352222-d31d3bdb4cc7.json"

if __name__=='__main__':
    vAR_hide_footer = """<style>
            footer {visibility: hidden;}
            </style>
            """
    vAR_st.markdown(vAR_hide_footer, unsafe_allow_html=True)
    try:
        # Applying CSS properties for web page
        CSS_Property("DSAI_Utility/DSAI_style.css")
        # Initializing Basic Componentes of Web Page
        All_Initialization()


        col1,col2,col3,col4,col5 = vAR_st.columns([1,9,1,9,2])
        with col2:
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.subheader('Select the Model')
            
        with col4:
            vAR_st.write('')
            vAR_option = vAR_st.selectbox(' ',('Select a Model',"Regression","Driver Risk - Crash Level Classification"))
            
            # vAR_option = vAR_st.selectbox('',('Select a Model',"Regression","Classification","Vertex AI - Regression (Deployed Model)","Vertex AI - Classification (Deployed Model)"))
            

        # if vAR_option=="Vertex AI - Regression (Deployed Model)":
        #     DriverRiskPrediction()
        
        # elif vAR_option=="Vertex AI - Classification (Deployed Model)":
        #     DriverRiskClassification()
            
        if vAR_option=="Regression":
            Regression_Model()
            
                        
        elif vAR_option=="Classification":
            Classification_Model()
            
        elif vAR_option=="Driver Risk - Crash Level Classification":
            DriverRiskClassification()
            
            # elif vAR_option=="Play with LLM-RAG(Retrieval Augmented Generation)":
            #     LLM_RAG()
            
        
    


    except KeyError as exception:
        print('Key Error - ',str(exception))
        # vAR_st.error(str(exception))
        pass
        

    except BaseException as exception:
        print('Error in main function - ', exception)
        exception = 'Something went wrong - '+str(exception)
        traceback.print_exc()
        vAR_st.error(exception)
