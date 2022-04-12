# File responsable for creating documents_df which contains the processed data, and saving it to documents_df_pickle.txt

import numpy as np
import pandas as pd
import pickle
from preprocessingFunctionModule import preprocess

subjects_df = pd.read_json("AuxFiles\\MechatronicsEngeneeringSubjects.json")
subjects_df = subjects_df.sort_values(by=["codigo"])
subjects_df = subjects_df.reset_index(drop=True)

documents_list = []

for i, row in subjects_df.iterrows():

    # * reading values of each subject (row)
    subject_id = row["codigo"]
    name = row["nome"]
    syllabus = row["ementa"]
    content = row["conteudo"]

    # * combining them to create the subject document
    text = name + ' ' + syllabus + ' ' + content
    
    # * preprocessing
    preProcessedText = preprocess(text)
    documents_list.append(preProcessedText)

documents_series = pd.Series(documents_list, name="documento")

documents_df = pd.concat([subjects_df, documents_series], axis=1)

with open('AuxFiles\\documents_df_pickle.txt', 'wb') as f:
    pickle.dump(documents_df, f)