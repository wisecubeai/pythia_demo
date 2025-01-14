import streamlit as st
import pandas as pd
import numpy as np
import json
from wisecube_sdk.client import WisecubeClient
from wisecube_sdk.model_formats import OutputFormat
import streamlit as st


hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """

st.markdown(hide_footer_style, unsafe_allow_html=True)

st.header('Pythia Hallucination Detection Demo')

@st.cache_data()
def create_client():
     client = WisecubeClient(st.secrets.api_key).client
     client.output_format=OutputFormat.PANDAS
     return client

@st.cache_data(show_spinner="Asking Pythia to detect Hallucinations...")
def ask_pythia(reference, response, question):
     client = create_client()
     # update the skd and fix signature on ask pythia ...
     output = client.ask_pythia(reference,response,question)
     metrics = output['data']['askPythia']['metrics']
     claims = output['data']['askPythia']['claims']
     return (claims, metrics)

     

@st.cache_data()
def load_examples(file_name):
     # Opening JSON file
     examples_file = open(file_name)
     # returns JSON object as a dictionary
     data = json.load(examples_file)
     return data


data = load_examples('examples.json')

num_examples = len(data['examples'])
# Create a list in a range of 1-num_examples
example_list = [*range(1, num_examples+1, 1)] 

option_string = st.selectbox(
     'Select an Example',example_list)

st.write('You Selected Example was Example:', option_string)
option = int(option_string)-1


reference_text = st.text_area("Reference Text", data['examples'][option]['reference'])
response_text = st.text_area("LLM Response", data['examples'][option]['response'])
question_text = st.text_area("Relevant Question if any", data['examples'][option]['question'])


if st.button('Ask Pythia', type="primary"):
     try:
         claims, metrics = ask_pythia(reference_text, response_text, question_text)
         st.subheader('Semantic claims extracted:')
         st.dataframe(claims)
    
         st.subheader('Pythia Accuracy Metrics:')
    
         st.write('Overall Accuracy:', str(float(metrics['accuracy'])*100.00)+"%") 
    
         contradictions=float(metrics['contradiction'])*100.00
         entailments=float(metrics['entailment'])*100.00
         neutrals=float(metrics['neutral'])*100.00
    
         df = pd.DataFrame(columns=['Percentage'])
         df.loc[len(df.index)] = [entailments]  
         df.loc[len(df.index)] = [contradictions]  
         df.loc[len(df.index)] = [neutrals]  
         df.index = ['Entailments', 'Contradiction', 'Neutrals']
         #st.write(df)
         st.bar_chart(df, color=["#fd0"])
     except Exception as e:
         if reference_text is None or response_text is None :
             st.error('Reference or Response is empty!')
         else :
             st.error('Something went wrong ...')
         




