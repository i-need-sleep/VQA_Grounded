import json
import glob

def eval(pred_path, true_path):

    with open(pred_path, 'r') as f:
        preds = json.load(f)
    
    with open(true_path, 'r') as f:
        trues_data = json.load(f)
    
    trues = []
    for key, val in trues_data.items():
        trues.append(val)

    score = 0

    for idx, true_instance in enumerate(trues):
        pred = preds[idx]['answer']

        correct = 0
        for true in true_instance['answers']:
            if true == pred:
                correct += 1
                if correct == 3:
                    break
        score += correct / 3

    score /= len(preds)    
    return score

if __name__ == '__main__':
    true_path = '../data/VisWis_VQA_Grounding/annotations/val_grounding.json'
    pred_root = '../output/vqa/result/*'
    # pred_root = '../output/vqa_mask/result/*'

    for pred_path in glob.glob(pred_root):
        
        if 'rank' in pred_path:
            continue
        
        score = eval(pred_path, true_path)
        print(pred_path)
        print(score)