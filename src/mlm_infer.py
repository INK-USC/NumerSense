from happytransformer import HappyXLNET
from happytransformer import HappyROBERTA
from happytransformer import HappyBERT
from tqdm import tqdm
import sys
import json
import os
#--------------------------------------#


num_list = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "no", "zero"]

if __name__ == "__main__":
    model_str = sys.argv[1]
    tsv_name = sys.argv[2]
    model = None
    if model_str.startswith("reload_"):
        if model_str.startswith("reload_bert"):
            model_str = model_str.replace("reload_roberta:", "")
            model = HappyBERT(model_str)
        elif model_str.startswith("reload_roberta"):
            model_str = model_str.replace("reload_roberta:", "")
            model = HappyROBERTA(model_str)
    else:
        if model_str.startswith("bert"):
            # bert-base, bert-large
            model = HappyBERT(model_str+"-uncased")
        elif model_str.startswith("roberta"):
            # roberta-base, roberta-large
            model = HappyROBERTA(model_str)
        elif model_str.startswith("xlnet"):
            # ignore
            model = HappyXLNET(model_str+"-cased")
    assert model is not None

    with open(tsv_name) as f:
        data = f.read().split("\n")
    output_tsv = []
    correct_cnt = 0
    correct_top2_cnt = 0
    correct_top3_cnt = 0
    correct_probs = 0.0
    for item in tqdm(data):
        masked_sent, truth = item.split("\t") 
        result_list = model.predict_mask(masked_sent, options=num_list, num_results=1000)
        result_list.sort(key=lambda x:x["softmax"], reverse=True)
        # cast float32 to float 
        result_list = [{"word":x["word"].lower(), "softmax":float(x["softmax"])} for x in result_list]
        result_word_list = [x["word"] for x in result_list]
        if len(result_list)>0:
            if truth == result_word_list[0]:
                correct_cnt += 1
            if truth in result_word_list[:2]:
                correct_top2_cnt += 1
            if truth in result_word_list[:3]:
                correct_top3_cnt += 1
        if truth in result_word_list:
            truth_index = result_word_list.index(truth)
            correct_probs += result_list[truth_index]["softmax"]
        score_str = json.dumps(result_list)
        output_tsv.append("\t".join([masked_sent, truth, score_str]))
    print("top1-acc:", correct_cnt/len(data))
    print("top2-acc:", correct_top2_cnt/len(data))
    print("top3-acc:", correct_top3_cnt/len(data))
    # print("mean-probs:", correct_probs/len(data))
    print("%.2f\t%.2f\t%.2f"%(correct_cnt/len(data)*100, correct_top2_cnt/len(data)*100, correct_top3_cnt/len(data)*100))
    tsv_name_base = os.path.basename(tsv_name)
    with open("pred_results/%s.%s.pred.tsv"%(tsv_name_base, model_str), 'w') as f:
        print("Saving resutls to", f.name)
        f.write("\n".join(output_tsv))
