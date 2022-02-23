# Collect answers that appears at least k tims in the dataset

import json

TRAIN_SET_PATH = '../data/VisWis_VQA_Grounding/annotations/train_grounding.json'
OUT_PATH = '../processed/answer_list.json'

def main(min_occurrence = 3):
    with open(TRAIN_SET_PATH, 'r') as f:
        data = json.load(f)

    answer_dict = {}
    
    for instance in data.keys():
        for answer in data[instance]['answers']:
            if answer in answer_dict:
                answer_dict[answer] += 1
            else:
                answer_dict[answer] = 1

    out = []
    for answer in answer_dict:
        if answer_dict[answer] >= min_occurrence:
            out.append(answer)

    print(f'Mininal #Occurrence: {min_occurrence}, length: {len(out)}')

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    return

if __name__ == '__main__':
    main()