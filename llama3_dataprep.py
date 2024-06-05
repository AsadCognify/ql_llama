import json
import pandas as pd

def json_file_writer(file_path,data):
    with open(file_path, 'w+') as f:
        json.dump(data, f)


def llama3_data_preparation(system_prompt,query_list,response_list,out_file_path):
  data = []
  for query,response in zip(query_list,response_list):
    data_point= f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt} <|eot_id|><|start_header_id|>user<|end_header_id|>
    {query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {response} <|eot_id|>'''
    data.append({"text":data_point})
  ready_to_train={"version": "0.1.0",
               'data':data}

  json_file_writer(out_file_path,ready_to_train)


## Only for consolidated_dev.csv
def loading_inference_file(inference_file_path):
  df = pd.read_csv(inference_file_path)
  #Converting data frame into dict
  inference_dict = df.to_dict(orient='records')
  return inference_dict

def formating_question(data_point):
  questtion = data_point['question'] + ' (1) '+str(data_point['opa']) + ' (2) '+str(data_point['opb']) + ' (3) '+str(data_point['opc']) + ' (4) '+str(data_point['opd'])
  return questtion

def correct_option_extractor(data_point):
  return data_point['cop']



################
# VARIABLES
################

# CSV file to load
inference_file_path = "consolidated_dev.csv"

# Save the output
out_file_path = "ready_to_train.json"

# 
system_prompt = "You are presented with the following multiple choice question. Think step by step and then select the best answer. just return the correct option with its number"

###################
#
##################

query_list = []
response_list = []
inference_dict = loading_inference_file(inference_file_path)

for i in range(len(inference_dict)):
    query_list.append( formating_question(inference_dict[i]) )
    response_list.append( correct_option_extractor(inference_dict[i]) )


# Create output training json
llama3_data_preparation( system_prompt,query_list,response_list,out_file_path )