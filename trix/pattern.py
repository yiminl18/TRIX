import json,sys,math,os
import numpy as np 
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

def get_root_path():
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    #print("Parent path:", parent_path)
    return parent_path

root_path = get_root_path()
sys.path.append(root_path)
from model import model 
from gurobipy import Model, GRB
model_name = 'gpt4o'
vision_model_name = 'gpt4vision'


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

def read_file(file):
    data = []
    with open(file, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Print the line (you can replace this with other processing logic)
            data.append(line.strip())
    return data

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def record_extraction(pss,predict_labels,pdf_path):
    #print(predict_labels)
    #only return the first record 
    stats = load_cands(pdf_path)
    first_key = 'null'
    phrases = {} #record id -> phrases
    rid = 1
    ps = []
    for p in pss:
        if(p == first_key):#check the first record
            phrases[rid] = ps
            ps = []
            rid += 1
        ps.append(p)
        if(p in predict_labels):
            #print(p)
            if(first_key == 'null'):
                first_key = p
                #print('first key:', first_key)
    
    return phrases, stats

def get_bb_path(extracted_file):
    file = extracted_file.replace('.txt','.json')
    return file 

def get_bb_phrase(phrase, c, phrases_bb):
    if(phrase not in phrases_bb):
        return (-1,-1,-1,-1)
    bbs = phrases_bb[phrase]
    if(c<len(bbs)):
        return bbs[c]
    return (-1,-1,-1,-1)

def get_bbdict_per_record(record_appearance, phrases_bb, phrases):
    #phrases: phrase -> a list of bounding box for all records 
    #record_appearance: phrase p->the number of appearances of p so far 
    #output: a dict. phrase -> a list of bounding box for the phrase in current record
    pv = {}
    non_dul_phrases = list(set(phrases))#remove duplicated phrases 
    for p in non_dul_phrases:
        c = phrases.count(p)
        if(p not in pv and p in phrases_bb):
            cur = record_appearance[p]
            lst = phrases_bb[p][cur: cur + c]
            pv[p] = lst
            record_appearance[p] = cur + c
    return record_appearance, pv

def outlier_detect(dis, threshold = 1):
    lst = []
    for id, d in dis.items():
        lst.append(d)
    
    mean = np.mean(lst)
    std = np.std(lst)
    
    outliers = []
    for value in lst:
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(value)
    if(len(outliers) == 0):
        return [], 0,0,[]

    cutoff = min(outliers)
    false_pairs = []
    for id, d in dis.items():
        if(d >= cutoff):
            false_pairs.append(id)
    #print(cutoff)
    sum = 0
    new_lst = []
    for val in lst:
        if(val >= cutoff):
            continue
        new_lst.append(val)
        sum += val
    new_mean = sum/(len(lst) - len(outliers))
    return false_pairs, cutoff, new_mean, new_lst

def is_outlier(lst, d):
    lst.append(d)
    threshold = 1  
    mean = np.mean(lst)
    std = np.std(lst)
    
    outliers = []
    for value in lst:
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(value)
    
    cutoff = min(outliers)
    #print(cutoff)
    if(d >= cutoff):#is an outlier
        return 1
    return 0

def bipartite_match(edges, num_nodes_A, num_nodes_B):

    adjacency_matrix = np.full((num_nodes_A, num_nodes_B), np.inf)

    for u, v, w in edges:
        adjacency_matrix[u, v] = w

    csr_adjacency_matrix = csr_matrix(adjacency_matrix)

    row_ind, col_ind = min_weight_full_bipartite_matching(csr_adjacency_matrix)

    weight = 0
    matching = {}

    # Iterate over the indices returned by the matching function
    for row, col in zip(row_ind, col_ind):
        matching[row] = (col, adjacency_matrix[row,col])
    return matching


def is_metadata(meta, val):
    if(val not in meta):
        return 0
    # Initialize the list to store the indices
    indices = []
    start = 0

    # Loop to find all occurrences of the substring
    while True:
        index = meta.find(val, start)
        if index == -1:
            break
        indices.append(index)
        start = index + 1

    for i in indices:
        l = i-1
        r = i+len(val)
        f1 = 0
        f2 = 0
        if(l>=0 and (meta[l] == ' ' or meta[l] == '\n' or meta[l] == ':')):
            f1 = 1
        if(r<len(meta) and (meta[r] == ' ' or meta[r] == '\n' or meta[r] == ':')):
            f2 = 1
        if(l<0):
            f1 = 1
        if(r == len(meta)):
            f2 = 1
        if(f1 == 1 and f2 == 1):
            return 1
    return 0


def key_val_extraction_by_first_learn(pv, predict_labels):
    #print(pv, predict_labels)
    kvs = []
    i = 0
    while i < len(pv):
        p = pv[i][0]
        if (p in predict_labels and i+1<len(pv) and pv[i+1][0] not in predict_labels):
            kvs.append((p,pv[i+1][0]))
            i += 2
        elif(p in predict_labels and i+1<len(pv) and pv[i+1][0] in predict_labels):
            kvs.append((p,'missing'))
            i += 1
        elif(p in predict_labels and i == len(pv)-1):
            kvs.append((p,'missing'))
            i += 1
        elif(p not in predict_labels):
            i += 1
        else:
            i += 1

    return kvs

def key_val_extraction(pv, predict_labels):
    #metadata = get_metadata().lower()
    metadata = []
    kv = {}#relative location id -> (key,val)
    kk = {}#relative location id -> (key,key)
    vv = {}#relative location id -> (val,val)
    ids = []
    dis = {}
    for i in range(len(pv)):
        p = pv[i][0]
        # if('medical' in p):
        #     print(p)
        bbp = pv[i][1]
        if(p in predict_labels):
            if(i < len(pv)-1 and pv[i+1][0] not in predict_labels):#kv pair
                pn = pv[i+1][0]
                kv[i] = (p,pn)
                ids.append(i)
                ids.append(i+1)
                bbpn = pv[i+1][1]
                dis[i] = min_distance(bbp,bbpn)
                # print(p,pn,dis[i])
                # print(bbp)
                # print(bbpn)
    #print(dis)

    outliers, cutoff, new_mean, new_lst = outlier_detect(dis)
    
    #second pass: scan for kk and vv 
    
    for id in outliers:
        if(is_metadata(metadata, pv[id][0]) == 0):
            #print('outlier: ', pv[id][0])
            kv[id] = (pv[id][0],'')
            ids.append(id)

    single_v = []
    i = 0
    while i < len(pv):
        p = pv[i][0]
        if(i in ids):#skip the kv pairs 
            i+=1
            continue
        if(p in predict_labels):
            if(i < len(pv)-1 and pv[i+1][0] in predict_labels):#kk pair
                pn = pv[i+1][0]
                kk[i] = (p,pn)
            elif(i == len(pv)-1):
                kv[i] = (p,'')
        else:
            if(i < len(pv)-1 and pv[i+1][0] not in predict_labels):#vv pair
                pn = pv[i+1][0]
                vv[i] = (p,pn)
            else:
                #vk? 
                single_v.append(i)
        i+=1
        
    
    bad_kv = []
    #process kk pair 
    for id, (p,pn) in kk.items():
        if(id in ids):#skip this pair since we don't want to modifty it
            continue
        
        
        if(len(new_lst) > 0 and is_outlier(new_lst, min_distance(pv[id][1],pv[id+1][1])) == False and pair_oracle(p,pn) == 1):#valid distance or semantically same or  pair_oracle(p,pn) == 1
            kv[id] = (p,pn)#insert into kv
            ids.append(id)
            ids.append(id+1)
            #update pn: remove in kv pair starting with pn 
            for id, (pi,pni) in kv.items():
                if(pi == pn):
                    bad_kv.append(id)
                    break
        else:
                #print('not match')
            if(is_metadata(metadata,p) == 0):
                #print('kk:', p)
                kv[id] = (p,'')
                ids.append(id)
        
            
    
    #process vv pair
    for id, (p,pn) in vv.items():
        if(id in ids):#skip this pair since we don't want to modifty it
            continue
        if(pair_oracle(p,pn) == 1):
            kv[id] = (p,pn)#insert into kv
            #print('matched')
            ids.append(id)
            ids.append(id+1)

    #process single v
    for i in single_v:
        v = pv[i][0]
        
        if(key_oracle(v) == 1 and is_metadata(metadata, v) == 0):
            #print('single v:', v)
            kv[i] = (v,'')

    kv_out = []
    for id, (p,pn) in kv.items():
        if(id in bad_kv):
            continue
        if((is_metadata(metadata, p) == 1 and len(p) > 3) and (is_metadata(metadata, pn) == 1 and len(pn) > 3)):
            #print(p,pn)
            continue
        kv_out.append((p,pn))
    #print(kv_out)
    return kv_out

def min_distance(bb1,bb2):
    # min(current_bbox[0], word['x0']),
    # min(current_bbox[1], word['top']),
    # max(current_bbox[2], word['x1']),
    # max(current_bbox[3], word['bottom'])
    lx = [abs(bb1[0]-bb2[0]), abs(bb1[0]-bb2[2]),abs(bb1[2]-bb2[0]),abs(bb1[2]-bb2[2])]
    ly = [abs(bb1[1]-bb2[1]), abs(bb1[1]-bb2[3]),abs(bb1[3]-bb2[1]),abs(bb1[3]-bb2[3])]
    x_min = min(lx)
    y_min = min(ly)
    return max(x_min,y_min)


def pair_oracle(left,right):
    instruction = 'The following two phrases are extracted from a table. ' 'Is ' + right + ' a possible value for the key word ' + left + '? Return only yes or no. '
    context = ''
    prompt = (instruction,context)
    response = model(model_name,prompt)
    #print(response)
    if('yes' in response.lower()):
        return 1
    return 0

def key_oracle(val):
    instruction = 'The following phrase is either a key word or value extracted from a table. ' 'If ' + val + ' a key word, return yes. If'  + val + ' is a value, return no. '
    context = ''
    prompt = (instruction,context)
    response = model(model_name,prompt)
    #print(response)
    if('yes' in response.lower()):
        return 1
    return 0
    

def get_bblist_per_record(record_appearance, phrases_bb, phrases):
    #phrases: phrase -> a list of bounding box for all records 
    #record_appearance: phrase p->the number of appearances of p so far 
    #output: a list of tuple. Each tuple:  (phrase, bounding box) for current record 
    pv = []
    appear = {}
    record = {}
    for p in phrases:
        c = phrases.count(p)
        cur = record_appearance[p]
        if(p in phrases_bb):
            lst = phrases_bb[p][cur: cur + c]
            if(p not in appear):
                appear[p] = 0
                record[p] = cur + c
            else:
                appear[p] = appear[p] + 1

            bb = lst[appear[p]]
            pv.append((p,bb))
    for p in phrases:
        if(p in phrases_bb):
            record_appearance[p] = record[p]
            
    return record_appearance, pv

def get_bb_per_record(record_appearance, phrases_bb, phrases):
    #phrases: phrase -> a list of bounding box for all records 
    #record_appearance: phrase p->the number of appearances of p so far 
    #output: a list of tuple. Each tuple:  (phrase, bounding box) for current record 
    pv = []
    non_dul_phrases = list(set(phrases))#remove duplicated phrases 
    for p in non_dul_phrases:
        c = phrases.count(p)
        if(p not in pv and p in phrases_bb):
            cur = record_appearance[p]
            lst = phrases_bb[p][cur: cur + c]
            #create tuple instances 
            for bb in lst:
                pv.append((p,bb))
            record_appearance[p] = cur + c
    return record_appearance, pv

def get_horizontal_mid(bb):
    return (bb[0] + bb[2])/2

def find_closest_value(val, lst):
    if(len(lst) == 0):
        return -1
    # Use a list comprehension to calculate the absolute differences
    closest_val = min(lst, key=lambda x: abs(x - val))
    return closest_val

def is_inclusive(b1,b2):
    #input: b1 and b2 are bounding box of two phrases 
    if(b1[2] < b2[0]):
        return 0
    if(b2[2] < b1[0]):
        return 0
    return 1

def is_aligned(b1,b2):
    if(b1[1] > b2[3]):
        return 0
    if(b1[3] < b2[1]):
        return 0
    return 1

def is_overlap_vertically(b1,b2):
    if(b1[2] < b2[0]):
        return 0
    if(b1[0] > b2[2]):
        return 0
    return 1

def hash_tuple(tuple):
    p = tuple[0]
    bb = tuple[1]
    return (p,bb[0],bb[1],bb[2],bb[3])

def find_rows(vg, key_mp, bbv):
    #key_mp: cluster_id -> key
    #input: a cluster dict. Cluster id -> a list of tuples. Each tuple:  (phrase, bounding box) 
    #output: row_id -> a list of tuples. Each tuple:  (phrase, bounding box) 
    row_id = 0
    row_mp = {}
    pb = []
    re_map = {}#tuple -> key cluster_id
    for id, tuples in vg.items():
        if(id not in key_mp):
            continue
        key = key_mp[id]
        #print(key, id)
        for t in tuples:
            #print(t)
            pb.append(t)
            re_map[hash_tuple(t)] = key
    for t in pb:
        pi = t[0]
        bi = t[1]
        is_match = 0
        for id, tuples in row_mp.items():#scan cluster
            for tt in tuples:
                pj = tt[0]
                bj = tt[1]
                #print(bi,bj, is_aligned(bi,bj))
                if(is_aligned(bi,bj) == 1):
                    #print(bi,bj)
                    is_match = 1
                    row_mp[id].append(t)
                    break
            if(is_match == 1):
                break
        if(is_match == 0):
            row_mp[row_id] = [t]
            row_id += 1

    # for id, lst in row_mp.items():
    #     print(id)
    #     for (p,bb,w) in lst:
    #         print(p)

    new_row_mp = {}
    keys = sort_keys(key_mp, bbv)
    row_loc = {}# x['top']-> id
    #sort row based on keys
    #print(len(row_mp))
    for id, tuples in row_mp.items():
        lst = []
        quick_mp = {}
        for t in tuples:
            key = re_map[hash_tuple(t)]
            quick_mp[key] = t
        x_top = 0
        for key in keys:
            if(key in quick_mp):
                tuple = quick_mp[key]
                lst.append(tuple)
                x_top = tuple[1][1]
            else:
                lst.append(('',[0,0,0,0]))#denote missing value 
        row_loc[x_top] = id
        new_row_mp[id] = lst

    # for id, lst in new_row_mp.items():
    #     print(id)
    #     #print(lst)
    #     for item in lst:
    #         print(item[0])

    #sort row based on bound box 
    sorted_rows = []
    sorted_row_loc = dict(sorted(row_loc.items()))
    #print(sorted_row_loc)
    for x_top, id in sorted_row_loc.items():
        sorted_rows.append(new_row_mp[id])

    return sorted_rows, keys




def sort_val_based_on_bb_width(pv, predict_labels):
    new_pv = []
    for item in pv:
        p = item[0]
        bb = item[1]
        width = bb[2] - bb[0]
        new_pv.append((p,bb,width))
    sorted_list = sorted(new_pv, key=lambda x: x[2])
    return sorted_list

def find_bb_value_group(vg):
    #input vg (val_group): cluster_id -> a list of tuples. Each tuple: (phrase, bb)
    #output: cluster_id -> bb of value group
    bbv = {}
    max = 10000
    min = 0
    for id, tuples in vg.items():
        b0 = max#x0
        b1 = max#top
        b2 = min#x1
        b3 = min#bottem
        for tuple in tuples:
            b = tuple[1]
            if(b[0] <= b0):
                b0 = b[0]
            if(b[2] >= b2):
                b2 = b[2]
            if(b[1] <= b1):
                b1 = b[1]
            if(b[3] >= b3):
                b3 = b[3]
        bbv[id] = (b0,b1,b2,b3)
    return bbv

def identify_headers(key_mp, predict_labels, footers):
    headers = []
    keys = []
    for id, key in key_mp.items():
        keys.append(key)
    for key in predict_labels:
        if(key not in keys and key not in footers):
            headers.append(key)
    return headers

def filter_key(bbv, pv, predict_labels):
    #output: a dict. cluster_id -> key
    vertical_dis = {}
    key_mp = {}
    for (phrase, b) in pv: 
        if(phrase in predict_labels):
            #print(phrase)
            #find the cloest key per value group
            for id, bb in bbv.items():
                if(is_inclusive(b,bb) == 1):
                    vdis = bb[1] - b[3]
                    if(id not in key_mp):
                        key_mp[id] = phrase
                        vertical_dis[id] = vdis
                    else:
                        if(vdis <= vertical_dis[id]):#update to the closest key 
                            key_mp[id] = phrase
                            vertical_dis[id] = vdis
    return key_mp
                    
def sort_keys(key_mp, bbv):
    #key_mp: cluster_id -> key
    #bbv: cluster_id -> bounding box of value group
    keys = []
    keys_bb = []
    for id, key in key_mp.items():
        bb = bbv[id]
        keys_bb.append((key,bb[0]))
    sorted_keys = sorted(keys_bb, key=lambda x: x[1])
    for key in sorted_keys:
        keys.append(key[0])
    return keys 

def find_value_group(pv, predict_labels):
    pv = sort_val_based_on_bb_width(pv, predict_labels)

    #input: a list of tuples. Each tuple:  (phrase, bounding box) 
    #output: a cluster dict. Cluster id -> a list of tuples
    # a tuple: (phrase, bounding box, width of bounding box)
    mp = {}
    id = 0
    footer = []
    for item in pv:
        #print(item)
        pi = item[0]
        if(pi in predict_labels):#skip keys, consider values 
            continue
        bbi = item[1]
        #print('***', pi,bbi)
        is_match = 0 
        matched_id = -1
        for i in range(id):#scan cluster 
            lst = mp[i]
            for pb in lst:
                pj = pb[0]
                bbj = pb[1]
                if(is_inclusive(bbi,bbj) == 1):
                    #add to current cluster 
                    # print(pi,pj)
                    # print(bbi,bbj)
                    #mp[i].append(item)
                    matched_id = i
                    is_match += 1
                    break
            if(is_match > 1):
                break
        if(is_match == 0):#there is no cluster matching with current item, create a new cluster
            mp[id] = [item]
            id += 1
        if(is_match == 1):
            mp[matched_id].append(item)
        if(is_match > 1):
            footer.append(item)
    return mp,footer

def h_distance(bb1, bb2):
    bb1_avg = (bb1[0]+bb1[2])/2
    bb2_avg = (bb2[0]+bb2[2])/2 
    return abs(bb1_avg - bb2_avg)

def key_val_mp(key_row, val_row):
    kv_mp = {}
    #print(val_row)
    #for each key, search its cloest and overlapping value 
    for item in key_row:
        key = item[0]
        keyb = item[1]
        hd = 10000000
        t_val = 'missing'
        for item in val_row:
            val = item[0]
            valb = item[1]
            if(is_overlap_vertically(keyb, valb) == 0):
                continue 
            if(h_distance(keyb,valb) < hd):
                hd = h_distance(keyb, valb)
                t_val = val
        kv_mp[key] = t_val
    #print(kv_mp)
    return kv_mp



def table_extraction_top_down(row_mp, kid, vid):
    key_row = row_mp[kid[0]]
    val_rows = []
    kvs = []
    rows = []# list of list 
    keys = []
    for (key,bb) in key_row:
        keys.append(key)
    #print(key_row)
    for id in vid:
        #print(id)
        val_rows.append(row_mp[id])
        #print(row_mp[id])
    for val_row in val_rows:
        kv = key_val_mp(key_row, val_row)
        #print(kv)
        kvs.append(kv)
    #clean the tabular format
    for kv in kvs:
        row = []
        for (key,bb) in key_row:
            row.append(kv[key])
        rows.append(row)
    #print(rows)
    return keys, rows


def table_extraction(predict_labels, pv, path):
    # for item in pv:
    #     print(item)
    vg,footer = find_value_group(pv, predict_labels)
    # for id, lst in vg.items():
    #     print(id)
    #     for item in lst:
    #         print(item[0])
        
    bbv = find_bb_value_group(vg)
    key_mp = filter_key(bbv, pv, predict_labels)
    headers = identify_headers(key_mp, predict_labels, footer)
    
    rows,keys = find_rows(vg, key_mp, bbv)


def print_table(keys, rows):
    keys_out = ', '.join(keys)
    keys_out += '\n'
    
    print(keys_out)
    for row in rows:
        row_out = ''
        for r in row:
            row_out += r[0] + ','
        print(row_out[:-1])

def write_table(keys, rows, path):
    keys_out = ', '.join(keys)
    keys_out += '\n'
    
    with open(path, 'w') as file:
        file.write(keys_out)
        #print(keys_out)
        for row in rows:
            row_out = ''
            for r in row:
                row_out += r[0] + ','
            file.write(row_out[:-1]+'\n')
            #print(row_out[:-1])

def format_dict(dict):
    d = {}
    for k,v in dict.items():
        d[k.lower()] = v
    return d

def write_result(results,path):
    with open(path, 'w', newline='') as file:
        # Write each row to the CSV file
        for row in results:
            key = row[0]
            val = row[1]
            page = 1.0
            if(',' in val):
                val = val.replace(',','')
            out = key +','+val +','+str(page)
            #print(out)
            file.write(out + '\n')

def is_same_row(b1,b2):
    #b2 should be in the right side of b1
    if(b2[0] < b1[0]):
        return 0
    # b1 and b2 should overlap in y
    if(b1[3] < b2[1] or b1[1] > b2[3]):
        return 0
    return 1

def major_overlapping_phrase(t,p1,p2):
    #choose between p1 and p2 which is overlapping more with t
    a=0 


def row_aligned(row1, row2, esp = 0.8):
    #check if there exist a phrase in row2 that overlapps with more than 2 phrases in row1
    
    id1 = 1 #id in row 1
    id2 = 0 #id in row 2
    match = 0
    #a value should not overlap with two keys
    while(id1 < len(row1) and id2 < len(row2)):
        if(is_overlap_vertically(row2[id2][1], row1[id1][1]) == 1 and is_overlap_vertically(row2[id2][1], row1[id1-1][1]) == 1):
            return 0
        #print(row1[id1][1][2], row2[id2][1][2])
        if(row1[id1][1][2] < row2[id2][1][2]):
            id1 += 1
        else:
            id2 += 1
    #a key should not overlap with two values: the reason is that some keys are seperated by a large phrase and thus the bounding box for a key might be fully accurate (larger than original) 
    #row1: key, row2: val
    # id1 = 0
    # id2 = 1
    # while(id1 < len(row1) and id2 < len(row2)):
    #     if(is_overlap_vertically(row1[id1][1], row2[id2][1]) == 1 and is_overlap_vertically(row1[id1][1], row2[id2-1][1]) == 1):
    #         return 0
    #     #print(row1[id1][1][2], row2[id2][1][2])
    #     if(row1[id1][1][2] < row2[id2][1][2]):
    #         id1 += 1
    #     else:
    #         id2 += 1

    #write the percetage of keys that have the correspoinding unique value mapping

    #check if there exist a phrase in row2 that does not overlapps with any of val in row1 - this is not robust: if there exist one phrase that violates this condition, then this row would be not correct 
    
    # for (p2,bb2) in row2:
    #     valid = 0
    #     for (p1,bb1) in row1:
    #         if(is_overlap_vertically(bb2,bb1) == 1):
    #             valid = 1
    #             break
    #     if(valid == 0):
    #         return 0
    return 1

def row_pattern_by_learned_pattern(lst, predicat_labels):
    is_key = 0
    is_val = 0
    #print(lst, predicat_labels)
    for (l,bb) in lst:
        if(l in predicat_labels):
            is_key = 1
        else:
            is_val = 1
    if(is_key == 1 and is_val == 0):
        return 'key'
    if(is_key == 0 and is_val == 1):
        return 'val'
    if(is_key == 1 and is_val == 1):
        return 'kv'

def row_pattern(lst, predict_labels, new_lst, esp = 0.5):
    kvs = 0
    kks = 0 
    vvs = 0
    
    p_pre = lst[0][0]
    bb_pre = lst[0][1]
    for i in range(1,len(lst)):
        p = lst[i][0]
        bb = lst[i][1]
        if(p_pre in predict_labels and p in predict_labels):
            kks += 1
        elif(p_pre in predict_labels and p not in predict_labels):
            #if(is_outlier(new_lst,min_distance(bb,bb_pre)) == 0):
                kvs += 1
                #print(p_pre,p)
            # else:
            #     vvs += 1
            #print(p_pre,p)
        elif(p_pre not in predict_labels and p not in predict_labels):
            vvs += 1
        p_pre = p
        
    size = kks + kvs + vvs
    #print(kks, kvs, vvs)
    if(size == 0):
        return 'undefined'
    
    if(kks / size > esp):
        return 'key'
    
    if(kvs / size > esp):
        return 'kv'
    
    if(vvs / size > esp):
        return 'val'
    
    return 'undefined'
    
def check_vadility(row_mp, rls, id): 
    if(rls[id] == 'undefined'):
        return 'undefined'
    
    if(rls[id] == 'key'):
        #for a 'key' row, if the next val is not aligned with key, make it to be undefined
        valid = 0
        nid = id
        while(True):
            nid += 1
            if(nid >= len(row_mp)):
                break
            if(rls[nid] == 'kv' or rls[nid] == 'key'):#mark the end of current table block
                #print('kv or key')
                break
            if(rls[nid] == 'val'):
                if(row_aligned(row_mp[id], row_mp[nid])==1):
                    valid = 1
                    break
                else:
                    valid = 0
                    break 
            # else:
            #     print('key-val not aligned ***')
        if(valid == 0):
            return 'undefined'
        else:
            return 'key'
        
    if(rls[id] == 'val'):
        #for a 'val' row, if there is no aligned key row above it, make it to be undefined
        valid = 0
        nid = id
        while(True):
            nid -= 1
            if(nid < 0):
                break
            if(rls[nid] == 'key' and row_aligned(row_mp[nid],row_mp[id]) == 1):
                valid = 1
                break
            if(rls[nid] == 'key' and row_aligned(row_mp[nid],row_mp[id]) == 0):
                break
            if(rls[nid] == 'kv'):
                break
        if(valid == 0):
            return 'undefined'
        else:
            return 'val'
    return rls[id]
        
def infer_undefined_LLM(lst):
    vals = []
    for (val,bb) in lst:
        vals.append(val)
    #evaluate by LLM
    context = ', '.join(vals)
    instruction = 'If the following list of phrases are mostly keywords, return key. If the following list of phrases are mostly values, return value. If the following list of phrases are a list of key value pairs where the key can have missing values, return kv. Do not add explanations, only return one of {key, value, kv}. '
    prompt = (instruction,context)
    response = model(model_name,prompt)
    #print(response)
    if('kv' in response):
        return 'kv'
    if('key' in response):
        return 'key'
    if('value' in response):
        return 'val'
    return 'undefined'
        
        

def infer_undefined(row_mp, rls):
    #guess the label of undefined rows based on rules 
    #try to modify undefined to different labels and check if this trial is valid or not 
    for row_id, lst in row_mp.items():
        if(rls[row_id] != 'undefined'):
            continue
        if(len(lst) > 1):
            #print('LLM call!', rls[row_id])
            #print(lst)
            label = infer_undefined_LLM(lst)
            #print(label)
            if(label == 'val'):
                rls[row_id] = 'val'
                if(check_vadility(row_mp, rls, row_id) == 'val'): #if undefined->val, it is valid
                    continue 
                else:
                    rls[row_id] = 'undefined'
            elif(label == 'key'):
                rls[row_id] = 'key'
                if(check_vadility(row_mp, rls, row_id) == 'key'): #if undefined->key, it is valid
                    continue
                else:
                    rls[row_id] = 'undefined'
            elif(label == 'kv'):
                rls[row_id] = 'kv'
            else:
                rls[row_id] = label

        elif(row_id-1>=0 and row_id+1 < len(row_mp) and rls[row_id-1] == 'kv' and rls[row_id+1] == 'kv'):
            rls[row_id] = 'kv' 
        elif(row_id-1>=0 and row_id+1 < len(row_mp) and rls[row_id-1] == 'val' and rls[row_id+1] == 'val'):
            #semantic check if it's a key inside a table block
            rls[row_id] = 'val'
            if(check_vadility(row_mp, rls, row_id) == 'val'):
                rls[row_id] = 'val'
            else:
                rls[row_id] = 'undefined'
        else:
            rls[row_id] = 'undefined'
    return rls

def check_kvs(rid, records, cans):
    while(rid-1 < len(cans)):
        records.append(cans[rid-1])
        rid += 1
    return records    
 
def pattern_detect_by_row(pv, predict_labels, rid, debug = 0):
    #refine kv pair by using distance constraint
    kv = {}
    ids = []
    dis = {}
    for i in range(len(pv)):
        p = pv[i][0]
        bbp = pv[i][1]
        if(p in predict_labels):
            if(i < len(pv)-1 and pv[i+1][0] not in predict_labels):#kv pair
                pn = pv[i+1][0]
                kv[i] = (p,pn)
                ids.append(i)
                ids.append(i+1)
                bbpn = pv[i+1][1]
                dis[i] = min_distance(bbp,bbpn)

    outliers, cutoff, new_mean, new_lst = outlier_detect(dis)
    # for i in outliers:
    #     print(kv[i])

    
    #input: a list of tuple. Each tuple:  (phrase, bounding box) for current record
    p_pre = pv[0][0]
    bb_pre = pv[0][1]
    row_id = 0
    row_mp = {} #row_id -> a list of (phrase, bb) in the current row
    row_mp[row_id] = []
    row_mp[row_id].append((p_pre, bb_pre))
    for i in range(1, len(pv)):
        p = pv[i][0]
        bb = pv[i][1]
        if(is_same_row(bb_pre,bb) == 0):
            row_id += 1
            row_mp[row_id] = []
        row_mp[row_id].append((p, bb))
        p_pre = p
        bb_pre = bb

    if(debug == 1):
        print('first pass label prediction')
    rls = {}
    for row_id, lst in row_mp.items():
        row = []
        for l in lst:
            row.append(l[0])
        if(rid == 1):
            row_label = row_pattern(lst, predict_labels, new_lst)
        else:
            row_label = row_pattern_by_learned_pattern(lst, predict_labels)
        if(debug == 1):
            print(row_id, row_label)
            p_print = []
            for (p,bb) in lst:
                p_print.append(p)
            print(p_print)
        rls[row_id] = row_label
        
    #if(rid == 1): #expensive learning for the first record
        #check the validity of the labels by using rules 
    if(debug == 1):
        print('second pass validation')
    for row_id, lst in row_mp.items():
        new_label = check_vadility(row_mp, rls, row_id)
        if(debug == 1):
            print(row_id)
            p_print = []
            for (p,bb) in lst:
                p_print.append(p)
            print(p_print)
            print(rls[row_id], new_label)
        rls[row_id] = new_label

    if(debug == 1):
        print('third pass infer undefined')
    rls = infer_undefined(row_mp, rls)

    if(debug == 1):
        for row_id, lst in row_mp.items():
            print(row_id, rls[row_id])
            p_print = []
            for (p,bb) in lst:
                p_print.append(p)
            print(p_print)
        
    blk, blk_id = block_decider(rls)
    
    return blk, blk_id, row_mp
    
def load_cands(pdf_path):
    if('benchmark1' in pdf_path):
        path = pdf_path.replace('data/raw','result').replace('.pdf','.json')
    else:
        path = pdf_path.replace('data/raw','result').replace('.pdf','_TWIX_kv.json')
    data = read_json(path)
    return data
    
def block_decider(rls):
    blk = {}#store the community of all rows belonging to the same block: bid -> a list of row id 
    blk_id = {}#store the name per block: bid-> name of block
    bid = 0
    nearest_key_bid = 0
    kv_bid = -1 #all kvs can be put into one block
    for id, label in rls.items():
        if(label == 'key'):
            bid += 1
            blk[bid] = []
            blk[bid].append(id)
            blk_id[bid] = 'table'
            nearest_key_bid = bid
        elif(label == 'val' and nearest_key_bid > 0):
            blk[nearest_key_bid].append(id)
        elif(label == 'kv'):#start a new block for kv
            if(kv_bid == -1):
                bid += 1
                kv_bid = bid
                blk[kv_bid] = []
            blk[kv_bid].append(id)
            blk_id[kv_bid] = 'kv'
    #block smooth for kv 
    #impute all the undefined row inside the kv block to be kv 
    for bid, name in blk_id.items():
        if(name == 'kv'):
            #print(blk[bid])
            new_row = []
            l = blk[bid][0]
            r = blk[bid][len(blk[bid])-1]
            for i in range(l,r+1):
                new_row.append(i)
            blk[bid] = new_row
            #print(blk[bid])
    return blk, blk_id

def write_json(out, path):
    with open(path, 'w') as json_file:
        json.dump(out, json_file, indent=4)

def filter_non_key(lst, non_key):
    nl = []
    for l in lst:
        if(l.lower() in non_key):
            continue
        nl.append(l)
    return nl

def mix_pattern_extract_pipeline(phrases_bb, predict_labels, phrases, path, pdf_path, debug = 0):
    phrases, stats = record_extraction(phrases, predict_labels, pdf_path)
    record_appearance = {}
    for rid, ps in phrases.items():
        for p in ps:
            record_appearance[p] = 0
    records = []
    rid = 1
    for rid, ps in phrases.items():
        record_appearance,pv = get_bblist_per_record(record_appearance, phrases_bb, ps)
        record = ILP_extract(predict_labels, pv, rid, stats)
        if(len(record) > 0):
            records.append(record)
    records = check_kvs(rid, records, stats)
    write_json(records, path)

def ILP_extract(predict_keys, pv, rid, recan):
    if(rid == 1):
        row_mp, row_labels = get_row_probabilities(predict_keys, pv)
        #LP formulation to learn row label assignment

        #pre-compute all C-alignments 
        Calign = {}
        for id1 in range(len(row_mp)):
            for id2 in range(id1+1, len(row_mp)):
                c = C_alignment(row_mp, id1, id2)
                Calign[(id1,id2)] = c
                Calign[(id2,id1)] = c
        
        row_pred_labels = ILP_formulation(row_mp, row_labels, Calign)

        #learn template
        blk, blk_id = template_learn(row_pred_labels)
        record, candidate = data_extraction(rid,blk,blk_id,row_mp,predict_keys,recan)
    else:
        if(rid-1<len(recan)):
            record = recan[rid-1]
        else:
            record = {}
    return record 

def ILP_formulation(row_mp, row_labels, Calign):
    model = Model("RT")
    #model.setParam('OutputFlag', 0)
    
    # create variables 
    # for each row, create four variables 
    vars = {} #row_id -> list of variables 
    for row_id, row in row_mp.items():
        var = []
        var_K = 'yK' + str(row_id)
        yk = model.addVar(vtype=GRB.INTEGER, name=var_K)
        var.append(yk)

        var_V = 'yV' + str(row_id)
        yV = model.addVar(vtype=GRB.INTEGER, name=var_V)
        var.append(yV)

        var_KV = 'yKV' + str(row_id)
        yKV = model.addVar(vtype=GRB.INTEGER, name=var_KV)
        var.append(yKV)

        var_M = 'yM' + str(row_id)
        yM = model.addVar(vtype=GRB.INTEGER, name=var_M)
        var.append(yM)

        vars[row_id] = var
    
    #add constraint 1: for each row, the sum of four variables is 1
    for row_id, var in vars.items():
        model.addConstr(var[0] + var[1] + var[2] + var[3] == 1, "SumOnePerRow")

    #add constraint 2: validity of key row
    for row_id, var in vars.items():
        operand = 0
        for j in range(row_id+1, len(vars)):
            operand += (Calign[(row_id,j)]*vars[j][1])
        model.addConstr(var[0] <= operand, "KeyValidity")

    #add constraint 3: validity of value row 
    for row_id, var in vars.items():
        operand = 0
        for j in range(0, row_id):
            operand += (Calign[(row_id,j)]*vars[j][0])
        model.addConstr(var[1] <= operand, "ValueValidity")

    #add optimization function
    log_prob = 0
    for row_id, var in vars.items():
        prob = var[0]*math.log(row_labels[row_id]['K'])
        prob += var[1]*math.log(row_labels[row_id]['V'])
        prob += var[2]*math.log(row_labels[row_id]['KV'])
        prob += var[3]*math.log(row_labels[row_id]['M'])
        log_prob += prob
    model.setObjective(log_prob, GRB.MAXIMIZE)
    
    # Optimize the model
    model.optimize()

    # get the predictions
    row_pred_labels = {}
    if model.status == GRB.OPTIMAL:
        label = ''
        for row_id, var in vars.items():
            if(var[0].x == 1):
                label = 'K'
            elif(var[1].x == 1):
                label = 'V'
            elif(var[2].x == 1):
                label = 'KV'
            elif(var[3].x == 1):
                label = 'M'
            row_pred_labels[row_id] = label
    
    return row_pred_labels

def template_learn(rls):
    blk = {}#store the community of all rows belonging to the same block: bid -> a list of row id 
    blk_id = {}#store the name per block: bid-> name of block
    bid = 0
    nearest_key_bid = 0
    kv_bid = -1 #all kvs can be put into one block
    for id, label in rls.items():
        if(label == 'K'):
            bid += 1
            blk[bid] = []
            blk[bid].append(id)
            blk_id[bid] = 'table'
            nearest_key_bid = bid
        elif(label == 'V' and nearest_key_bid > 0):
            blk[nearest_key_bid].append(id)
        elif(label == 'KV'):#start a new block for kv
            if(kv_bid == -1):
                bid += 1
                kv_bid = bid
                blk[kv_bid] = []
            blk[kv_bid].append(id)
            blk_id[kv_bid] = 'kv'
    #block smooth for kv 
    #impute all the undefined row inside the kv block to be kv 
    for bid, name in blk_id.items():
        if(name == 'kv'):
            #print(blk[bid])
            new_row = []
            l = blk[bid][0]
            r = blk[bid][len(blk[bid])-1]
            for i in range(l,r+1):
                new_row.append(i)
            blk[bid] = new_row
            #print(blk[bid])
    return blk, blk_id

def location_alignment(row_mp, id1, id2):
    return row_aligned(row_mp[id1], row_mp[id2]) 

def semantic_alignment(row_mp, id1, id2):
    pairs = []
    for cell1 in row_mp[id1]:
        val1 = cell1[0]
        bb1 = cell1[1]
        for cell2 in row_mp[id2]:
            val2 = cell2[0]
            bb2 = cell2[1]
            if(is_overlap_vertically(bb1,bb2) == 1):
                pairs.append((val1,val2))
                break
    #print(id1, id2, pairs)
    #construct prompts to LLMs
    instruction = 'Given a list of phrase pairs as (phrase1, phrase 2), for each pair of phrase, if two phrases are a key-value pair, return yes. Otherwise, return no. Do not add any explanations, only return yes or no for each phrase pair.'
    context = ", ".join(f"({repr(key)}, {repr(value)})" for key, value in pairs)
    prompt = (instruction,context)
    response = model(model_name,prompt)
    #print(prompt)
    # print(id1, id2)
    #print(response) 
    pos = response.lower().count('yes')
    if(len(pairs) == 0):
        return 0
    return pos/len(pairs)

def C_alignment(row_mp, id1, id2): #comprehensive alignment
    #print(id1,id2)
    l_score = location_alignment(row_mp, id1, id2)
    return l_score
    # print('l_score:', l_score)
    # if (l_score == 1):
    #     return 1
    # s_score = semantic_alignment(row_mp, id1, id2)
    # print('s_score:', s_score)
    # if(s_score >= 0.5):
    #     return 1
    # else:
    #     return 0

def get_row_probabilities(predict_keys, pv):
    #input: a list of tuple. Each tuple:  (phrase, bounding box) for current record

    #create row representations 
    p_pre = pv[0][0]
    bb_pre = pv[0][1]
    row_id = 0
    row_mp = {} #row_id -> a list of (phrase, bb) in the current row
    row_mp[row_id] = []
    row_mp[row_id].append((p_pre, bb_pre))
    for i in range(1, len(pv)):
        p = pv[i][0]
        bb = pv[i][1]
        if(is_same_row(bb_pre,bb) == 0):
            row_id += 1
            row_mp[row_id] = []
        row_mp[row_id].append((p, bb))
        p_pre = p
        bb_pre = bb
    
    #print row representation
    #print_rows(row_mp)

    #for each row, compute the probability per label
    row_labels = {} #row_id -> label, label is also a dict: label instance -> probability 
    for row_id, row in row_mp.items():
        row_labels[row_id] = row_label_prediction(row, predict_keys)
        
    #print_rows(row_mp, row_labels)
    return row_mp, row_labels

def row_label_prediction(row, predict_keys):
    kvs = 0
    kks = 0 
    vvs = 0
    
    p_pre = row[0][0]
    for i in range(1,len(row)):
        p = row[i][0]
        if(p_pre in predict_keys and p in predict_keys):
            kks += 1
        elif(p_pre in predict_keys and p not in predict_keys):
                kvs += 1
        elif(p_pre not in predict_keys and p not in predict_keys):
            vvs += 1
        p_pre = p

    total = kvs + kks + vvs
    label = {}
    delta = 0.001
    if(total == 0):
        label['K'] = delta
        label['V'] = delta
        label['KV'] = delta
        label['M'] = delta
    else:
        label['K'] = kks/total+delta
        label['V'] = vvs/total+delta
        label['KV'] = kvs/total+delta
        label['M'] = 2*delta 
    if(label['K'] == label['KV']):
        label['KV'] += delta/2
    return label

def print_rows(row_mp, row_labels):
    for row_id, lst in row_mp.items():
            print(row_id, row_labels[row_id])
            p_print = []
            for (p,bb) in lst:
                p_print.append(p)
            print(p_print)

def data_extraction(rid,blk,blk_id,row_mp,predict_labels,recan):
    out = []
    record = {}
    record['id'] = rid
    if(rid-1 < len(recan)):
        re = recan[rid-1]
    else: 
        re = {}

    for id, lst in blk.items():
        object = {}
        if(blk_id[id] == 'table'):
            #print(id)
            object['type'] = 'table'
            #print(blk_id[id],lst)#lst is the list of row ids belonging to the same community
            key = [lst[0]]
            vals = []
            for id in range(1,len(lst)):
                vals.append(lst[id])
            key, rows = table_extraction_top_down(row_mp, key, vals)
            content = []
            
            for row in rows:
                kvs = {}
                for i in range(len(key)):
                    k = key[i]
                    r = row[i]
                    kvs[k] = r
                content.append(kvs)
            #print(content)
            object['content'] = content 
        else:
            object['type'] = 'kv'
            #print(blk_id[id],lst)
            kvs = []#kvs stores a list of tuples, where each tuple is (phrase, bb)
            for id in lst:
                kvs += row_mp[id]
                kv_out = key_val_extraction_by_first_learn(kvs, predict_labels)
            content = []
            
            for kv in kv_out:
                kvm = {}
                if(kv[1] == ''):
                    kvm[kv[0]] = 'missing'
                else:
                    kvm[kv[0]] = kv[1]
                content.append(kvm)
            object['content'] = content
        out.append(object)
    
    record['content'] = out

    return re, record 

def mix_pattern_extract(predict_labels, pv, rid, debug = 0):
    
    #pv: a list of tuple. Each tuple:  (phrase, bounding box) for current record 
    keys = []

    blk, blk_id, row_mp = pattern_detect_by_row(pv, predict_labels, rid, debug)

    # print(blk)
    # print(blk_id)
    
    out = []
    record = {}
    record['id'] = rid

    for id, lst in blk.items():
        object = {}
        if(blk_id[id] == 'table'):
            #print(id)
            object['type'] = 'table'
            #print(blk_id[id],lst)#lst is the list of row ids belonging to the same community
            key = [lst[0]]
            vals = []
            for id in range(1,len(lst)):
                vals.append(lst[id])
            key, rows = table_extraction_top_down(row_mp, key, vals)
            content = []
            
            for row in rows:
                kvs = {}
                for i in range(len(key)):
                    k = key[i]
                    r = row[i]
                    kvs[k] = r
                    keys.append(k)
                content.append(kvs)
            #print(content)
            object['content'] = content 
        else:
            object['type'] = 'kv'
            #print(blk_id[id],lst)
            kvs = []#kvs stores a list of tuples, where each tuple is (phrase, bb)
            for id in lst:
                kvs += row_mp[id]
            if(rid == 1):
                kv_out = key_val_extraction(kvs, predict_labels)
            else:
                kv_out = key_val_extraction_by_first_learn(kvs, predict_labels)
            content = []
            
            for kv in kv_out:
                kvm = {}
                if(kv[1] == ''):
                    kvm[kv[0]] = 'missing'
                else:
                    kvm[kv[0]] = kv[1]
                keys.append(kv[0])
                content.append(kvm)
            object['content'] = content
        out.append(object)
    
    record['content'] = out

    return record,keys 
    
        

def write_string(result_path, content):
    with open(result_path, 'w') as file:
        file.write(content)

def get_extracted_path(path, method = 'plumber'):
    path = path.replace('raw','extracted')
    if('benchmark1' in path):
        path = path.replace('.pdf', '_' + method +  '.txt')
    else:
        path = path.replace('.pdf', '.txt')
    return path

def template_based_data_extraction(pdf_path, out_path):
    key_path = pdf_path.replace('data/raw','out').replace('.pdf','_TWIX_key.txt')
    extracted_path = get_extracted_path(pdf_path)
    
    if(not os.path.isfile(extracted_path)):
        return 
    if(not os.path.isfile(key_path)):
        return 
    bb_path = get_bb_path(extracted_path)
    
    
    keywords = read_file(key_path)#predicted keywords
    phrases = read_file(extracted_path)#list of phrases
    phrases_bb = read_json(bb_path)#phrases with bounding boxes
    debug_mode = 0

    print('Template-based data extraction starts...')

    mix_pattern_extract_pipeline(phrases_bb, keywords, phrases, out_path, pdf_path, debug_mode)

