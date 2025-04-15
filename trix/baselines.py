import sys,os
import key,eval,time
current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(root_path)
from model import model 
from pdf2image import convert_from_path
model_name = 'gpt4o'
vision_model_name = 'gpt4vision'

def LLM_key_extraction(phrases, path):
    instruction = 'The following list contains keys and values extracted from a document, return all the keys seperated by |. Do not generate duplicated keys. Do not make up new keys.'
    delimiter = ', '
    context = delimiter.join(phrases)
    prompt = (instruction,context)
    response = model(model_name,prompt)
    #print(response)
    l = response.split('|')
    key.write_result(path, l)

def LLM_table_extraction(phrases, path):
    instruction = 'The following list contains keys and values extracted from a table, return the table. The first row is a set of keys, the following rows are the values corresponding to the keys, seperate keys and values by using comma. '
    delimiter = ', '
    context = delimiter.join(phrases)
    prompt = (instruction,context)
    response = model(model_name,prompt)
    #print(response)
    #l = response.split('|')
    key.write_result(path, response)

def LLM_KV_extraction(phrases, path):
    instruction = 'The following data contains keys and values extracted from a document, find all the keys and their corresponding values. In each line, return key,value. Do not make up new phrase.'
    delimiter = ', '
    context = delimiter.join(phrases)
    #print(context)
    prompt = (instruction,context)
    response = model(model_name,prompt)
    #print(response)
    key.write_result(path, response)

def LLM_mix_extraction(phrases, path):
    # instruction = 'This document contains tables or key values. If it contains tables, return a line of table schema and seperate phrases by using comma. For each row of the table, display the row in a line and  seperate phrases by using comma. If it contains key values, return a list of key values in the reading order, where in each line, use the format of key:value to show a pair of key and value. If it contains both tables and key values, extract the content of table and key values in the above format. Do not add explanations.'
    instruction = 'This document might contain tables or key values. Process the document in the reading order. If it has a table block, output a json object called table. In table object, each column is a key and all the corresponding values are the values of the key. If it has a key value block, output a json object called KV. In KV object, output each pair of key value to be the key and value in the json object. Do not add explanations and do not make up new prhases. Return the output in the json format. '
    delimiter = ', '
    context = delimiter.join(phrases)
    #print(context)
    prompt = (instruction,context)
    response = model(model_name,prompt)
    #print(response)
    key.write_raw_response(path, response)

def pdf_2_image(path, page_num):
    images = convert_from_path(path, first_page = 1, last_page = page_num)
    for i in range(page_num):
        page_path = key.get_extracted_image_path(path, i)
        print(i)
        images[i] = images[i].save(page_path)
    return images

def LLM_KV_extraction_vision(image_path, out_path):
    # instruction = 'This document contains tables or key values. If it contains tables, return a line of table schema and seperate phrases by using comma. For each row of the table, display the row in a line and  seperate phrases by using comma. If it contains key values, return a list of key values in the reading order, where in each line, use the format of key:value to show a pair of key and value. If it contains both tables and key values, extract the content of table and key values in the above format. Do not add explanations.' 
    instruction = 'The document contains structure information, like tables or key values. Can you extract these information out? If it contains tables, return a line of table schema and seperate phrases by using comma. For each row of the table, display the row in a line and  seperate phrases by using comma. If it contains key values, return a list of key values in the reading order, where in each line, use the format of key:value to show a pair of key and value. If it contains both tables and key values, extract the content of table and key values in the above format. Do not add explanations.'
    context = ''
    prompt = (instruction,context)
    response = model(vision_model_name,prompt, image_path = image_path)
    #print(response)
    key.write_raw_response(out_path, response)
    return response

    