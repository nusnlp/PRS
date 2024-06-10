import sys
import json
import numpy as np
import jsonlines

def save_data_to_json(strings, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)

def id_to_item(path):
    id_to_item = {}
    with open(path, 'r') as f:
        content = f.readlines()
        for item in content:
            item = json.loads(item)
            id_to_item[item['id']] = item
    return id_to_item

d1 = id_to_item(sys.argv[1]) 
d2 = id_to_item(sys.argv[2])

items_to_save = []

for k1, v1 in d1.items():
    if k1 in d2:
        v2 = d2[k1]
        
        scores = v1['scores'] + v2['scores']
        responses = v1['responses'] + v2['responses']
        best_response = responses[np.argmax(scores)]

        new_item = v1
        new_item['responses'] = responses
        new_item['scores']  = scores
        new_item['score'] = np.max(scores)
        new_item['best_response'] = best_response    
        items_to_save.append(new_item)
    
    else:
        continue

save_data_to_json(items_to_save, sys.argv[3])
