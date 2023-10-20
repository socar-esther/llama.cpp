import os
import json

import pandas as pd
import numpy as np

from sklearn import metrics
from scipy.spatial.distance import cdist
from langchain.embeddings import LlamaCppEmbeddings

# Use Llama model for embedding
llama_model_path = './models/llama-2-7b/ggml-model-f16.gguf'
embeddings = LlamaCppEmbeddings(model_path=llama_model_path, n_ctx = 8192) 

# Get Query
task2_df = pd.read_csv('../../llama/dataset/task2_all.csv')

# Get Answer Vector
with open('../../llama/dataset/answer_db.json', 'r') as fp:
    ans_db = json.load(fp)
    
# Anser embedding을 여기서 먼저 구워둔다
answer_embedding_list = list()
for a_key, a_value in ans_db.items():
    tmp_a_embedding = embeddings.embed_query(a_value)
    answer_embedding_list.append(tmp_a_embedding)
pd.DataFrame(answer_embedding_list).to_csv('../../llama/dataset/answer_embedding.csv', index=False)

retrieved_answer_list = list()

for idx in range(len(task2_df)):
    
    if idx % 5 == 0:
        print(f'[LOG] {idx}/{len(task2_df)}')
        pd.DataFrame(retrieved_answer_list).to_csv('../../llama/dataset/task2_result.csv', index=False)
        
    row = task2_df.iloc[idx]
    
    # query부터 가져온다
    question = str(row['question'])
    q_embedding = embeddings.embed_query(question)
    
    # query랑 비교할 answer DB 부터 가져와서 확인하기
    key_list = list()
    sim_list = list()
    
    for a_key, tmp_a_embedding in enumerate(answer_embedding_list):
        # print(f'[DEBUG] AnsDB instacne: {a_value}')
        tmp_sim = 1 - cdist(np.array([tmp_a_embedding]), np.array([q_embedding]), metric='cosine')
        sim_list.append(tmp_sim)
    
    # 모든 Answer에 대한 sim score 저장하기 (for metric 계산)
    retrieved_answer_list.append(sim_list)

task2_df['pred'] = retrieved_answer_list
task2_df.to_csv('../../llama/dataset/task2_result.csv', index=False) # 최종 결과 저장