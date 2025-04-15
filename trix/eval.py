#this script implements the evaluation metric 
import json,os,math,csv
import Levenshtein

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data 
    
def get_leaf_nodes_paris(data):
    #this function get the key-val pairs in all leaves per record 
    kvs_record = {} #record id -> a list of kvs 
    keys = []
    for record in data:
        id = record['id']
        content = record['content']
        kvs = []
        for block in content:
            #print(block['type'])
            #skip the evaluation fo metadata for now
            if(block['type'] == 'metadata'):
                continue
            for tuple in block['content']:
                #print(tuple)
                for k,v in tuple.items():
                    keys.append(k)
                    kvs.append((k,v))
        kvs_record[id] = kvs
    keys = list(set(keys))
    return kvs_record, keys

import random

def write_json(out, path):
    with open(path, 'w') as json_file:
        json.dump(out, json_file, indent=4)

def clean_phrase(p):
    if(isinstance(p,str)):
        return p.lower().strip()
    return p

def can_convert_to_int(string):
    try:
        # Try converting the string to an integer
        int(string)
        return True
    except ValueError:
        # If a ValueError occurs, the string can't be converted to an integer
        return False

def can_convert_to_float(string):
    try:
        # Try converting the string to an integer
        float(string)
        return True
    except ValueError:
        # If a ValueError occurs, the string can't be converted to an integer
        return False
    
def normalize_string(s):
    # Replace literal '\\n' (backslash followed by 'n') and spaces, then convert to lowercase
    s = s.replace('\\n', '').replace(' ', '').lower()
    s = s.replace('|','').replace('\\','')
    return s

def approx_equal(str1, str2, esp = 0.95):
    # Calculate the Levenshtein distance (edit distance) between two strings
    distance = Levenshtein.distance(normalize_string(str1), normalize_string(str2))
    ratio = 1 - distance/(max(len(str1),len(str2)))
    if(ratio > esp):
        return 1
    else:
        return 0
def equal(a,b):
    if(isinstance(a,str) and isinstance(b,int) and can_convert_to_int(a) and int(a) == b):
        return 1
    if(isinstance(b,str) and isinstance(a,int) and can_convert_to_int(b) and int(b) == a):
        return 1
    if(isinstance(a,float) and isinstance(b,str) and can_convert_to_float(b) and a==float(b)):
        return 1
    if(isinstance(b,float) and isinstance(a,str) and can_convert_to_float(a) and float(a)==b):
        return 1
    if(a==b):
        return 1
    if(a=='missing' and b == ''):
        return 1
    if(a == '' and b == 'missing'):
        return 1
    if(a=='[missing]' and b == ''):
        return 1
    if(a == '' and b == '[missing]'):
        return 1
    if(isinstance(b, float) and math.isnan(b) and isinstance(a, str) and a.lower() == 'n/a'):
        return 1
    if(isinstance(a, float) and math.isnan(a) and isinstance(b, str) and b.lower() == 'n/a'):
        return 1
    if(isinstance(a,str)):
        a = a.strip('\'')
        a = a.strip('\"')
    if(isinstance(b,str)):
        b = b.strip('\'')
        b = b.strip('\"')
    if(isinstance(a,str) and isinstance(b,int)):
        if(str(b) == a):
            return 1
    if(isinstance(b,str) and isinstance(a,int)):
        if(str(a) == b):
            return 1
    if(isinstance(a,str) and isinstance(b,str)):
        if(a == b):
            return 1
        if(normalize_string(a) == normalize_string(b)):
            return 1
        if(approx_equal(a,b) == 1):
            return 1
    if(isinstance(a,str) and isinstance(b,str)):
        if(a.lower() == b.lower()):
            return 1
        #the rules below remove the errors from the OCR phrase extraction
        #the OCR phrase extraction should not be counted as the algorithm errors
        if(a.lower() in b.lower() and len(b) > 10):
            return 1
        if(b.lower() in a.lower() and len(a) > 10):
            return 1
    OCR_phrase = ['(defpelaorntmy cenotn vfiicntdioinngs)','department finding','m(adkeinpga rftamlseen st tfaitnedminegn)ts','(dfeeplaorntym ceonnt vfiicntdioinng)','tamperfiningd win/ge)vidence']
    for p in OCR_phrase:
        if(isinstance(a,str) and p in a.lower()):
            return 1
        if(isinstance(b,str) and p in b.lower()):
            return 1
        
    if(a == True and b == '\uf0fc'):
        return 1
    if(b == True and a == '\uf0fc'):
        return 1
    if(isinstance(a,str) and isinstance(b,str)):
        if(a.rstrip('.') == b.rstrip('.')):#remove tailing ..
            return 1
    if(isinstance(a,str) and isinstance(b,float)):
        if(can_convert_to_float(a.rstrip('.')) and float(a.rstrip('.')) == b):
            return 1
    if(isinstance(b,str) and isinstance(a,float)):
        if(can_convert_to_float(b.rstrip('.')) and float(b.rstrip('.')) == a):
            return 1
    return 0

def can_convert_to_float(s):
    try:
        float(s)  # Attempt to convert the string to a float
        return True
    except ValueError:
        return False

def merge_KVs(kvs):
    kvl = []
    for id, kv in kvs.items():
        for o in kv:
            kvl.append(o) 
    kvd = {}
    kvd[1] = kvl
    return kvd

def get_PR(results_kvs, truth_kvs):
    #print(len(truth_kvs))
    precisions = {} #record id ->  precision 
    recalls = {} # record id -> recall
    avg_precision = 0
    avg_recall = 0
    for id, truth_kv in truth_kvs.items():
        #print(id)
        precision = 0
        recall = 0
        if id not in results_kvs:
            precisions[id] = 0
            recalls[id] = 0
            continue
        result_kv = results_kvs[id]

        #clean phrases in results and truths
        new_truth_kv = []
        new_result_kv = []
        for kv in truth_kv:
            new_truth_kv.append((clean_phrase(kv[0]), clean_phrase(kv[1])))
        for kv in result_kv:
            new_result_kv.append((clean_phrase(kv[0]), clean_phrase(kv[1])))

        for kv in new_result_kv:
            is_match = 0
            for kv1 in new_truth_kv:
                if(equal(kv[0],kv1[0]) == 1 and equal(kv[1],kv1[1]) == 1):
                    precision += 1
                    is_match = 1
                    break
        
        if(len(new_result_kv) == 0):
            precision = 0
        else:
            precision /= len(new_result_kv)

        for kv in new_truth_kv:
            is_match = 0
            for kv1 in new_result_kv:
                if(equal(kv[0],kv1[0]) == 1 and equal(kv[1],kv1[1]) == 1):
                    recall += 1
                    is_match = 1
                    break
        recall /= len(new_truth_kv)
        

        precisions[id] = precision
        recalls[id] = recall 
    
    #compute the average 
    avg_precision = 0
    avg_recall = 0
    for id, precision in precisions.items():
        avg_precision += precision
    avg_precision /= len(precisions)
    for id, recall in recalls.items():
        avg_recall += recall
    avg_recall /= len(recalls)

    return avg_precision, avg_recall

def scan_folder(path, filter_file_type = '.json'):
    file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)
            if('DS_Store' in file_name):
                continue
            if(filter_file_type not in file_name):
                continue
            file_names.append(file_name)
    return file_names

def get_result_path(truth_path):
    result_path = ''
    result_path = truth_path.replace('')
    return result_path

def eval_one_doc(truth_path, result_path):
    truth = read_json(truth_path)
    truth_kvs, truth_keys = get_leaf_nodes_paris(truth)
    if('llmns_' not in result_path):
        result = read_json(result_path)
        result_kvs, result_keys = get_leaf_nodes_paris(result)
    else:
        result_kvs = get_kv_pairs_csv(result_path)
    #print(truth_kvs)
    avg_precision, avg_recall = get_PR(result_kvs, truth_kvs)
    print('precision:', avg_precision, 'recall:', avg_recall)
    return avg_precision, avg_recall

def get_kv_pairs_csv(result_path):
    kvs = {}
    with open(result_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        first_row = 0
        # Iterate over each row in the CSV
        for row in reader:
            record_id = row[0]
            if(record_id == 'Record'):
                continue
            #print(record_id)
            record_id = int(record_id)
            key = row[1].strip('"')
            value = row[2].strip('"')
            if(record_id not in kvs):
                kvs[record_id] = [(key,value)]
            else:
                kvs[record_id].append((key,value))
    return kvs 


def get_key_val_path(raw_path, approach):
    path = raw_path.replace('data/raw','result')
    path = path.replace('.pdf', '_' + approach + '_kv.json')
    return path

def get_baseline_result(raw_path, approach):
    path = raw_path.replace('data/raw','result')
    path = path.replace('.pdf','.csv') 
    file_name = path.split('/')[-1]
    file_name = approach + '_' + file_name
    directory_path = path.rsplit('/', 1)[0]
    new_path = directory_path + '/' + file_name
    return new_path

import os

def eval(approach):
    root_path = get_root_path()
    pdf_folder_path = root_path + '/data/raw'
    pdfs = scan_folder(pdf_folder_path,'.pdf')
    precision = 0
    recall = 0
    cnt = 0 
    
    for pdf_path in pdfs:
        result_path = ''
        if approach == 'TRIX':
            result_path = pdf_path.replace('data/raw','out').replace('.pdf','_TWIX_kv.json')
        if approach == 'Evaporate-Direct': 
            result_path = pdf_path.replace('data/raw','out').replace('.pdf','_Evaporate_Direct_kv.json')
        #print(result_path)
        if(not os.path.isfile(result_path)):
            #print('result not exsist:', result_path)
            continue 

        truth_path = pdf_path.replace('raw','truths').replace('.pdf','.json')
        #print(truth_path)
        if(not os.path.isfile(truth_path)):
            #print('truth not exist:', truth_path)
            continue 
        pdf_file_name = pdf_path.split('/')[-1]
        print(pdf_file_name)
        cnt += 1
        
        avg_precision, avg_recall = eval_one_doc(truth_path, result_path)
        precision += avg_precision
        recall += avg_recall
    precision /= cnt
    recall /= cnt
    print('average precision:', precision)
    print('average recall:', recall) 
        
def write_list(path, phrases):
    out = ''
    for phrase in phrases:
        out += phrase
        out += '\n'
    with open(path, 'w') as file:
        # Write the string to the file
        file.write(out)

def get_root_path():
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    #print("Parent path:", parent_path)
    return parent_path

def read_file(file):
    data = []
    with open(file, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Print the line (you can replace this with other processing logic)
            data.append(line.strip())
    return data

def match_phrases(keys, phrases):
    k = []
    for key in keys:
        for phrase in phrases:
            if(key.lower() == phrase.lower()):
                k.append(phrase)
                break
    return k

def load_keys():
    root_path = get_root_path()

    pdf_folder_path = root_path + '/data/raw/benchmark1'
    pdfs = scan_folder(pdf_folder_path,'.pdf')
    for pdf_path in pdfs:
        print(pdf_path)
        truth_path = pdf_path.replace('raw','truths').replace('.pdf','.json')
        if(not os.path.isfile(truth_path)):
            continue 
        extracted_path = pdf_path.replace('raw','extracted').replace('.pdf','.txt')

        if(not os.path.isfile(extracted_path)):
            continue

        phrases = read_file(extracted_path)
        truth = read_json(truth_path)
        truth_kvs, keys = get_leaf_nodes_paris(truth)
        keys = match_phrases(keys, phrases)
        target_path = pdf_path.replace('data/raw','result').replace('.pdf','_key.txt')
        print(keys)
        print(target_path)
        write_list(target_path, keys)
        

    

        

    