from happytransformer import HappyXLNET
from happytransformer import HappyROBERTA
from happytransformer import HappyBERT
from tqdm import tqdm
import sys
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
#--------------------------------------#


num_list = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "no", "zero"]

if __name__ == "__main__":
    model_str = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]
    cuda = False
    model = None
    if model_str.startswith("reload_"):
        if model_str.startswith("reload_bert"):
            model = HappyBERT(model_str.replace("reload_bert:", ""))
        elif model_str.startswith("reload_roberta"):
            model = HappyROBERTA(model_str.replace("reload_roberta:", ""))
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
        elif model_str.startswith("gpt"):
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model.eval()
            if cuda:
                model.to('cuda')

    assert model is not None

    with open(input_filename) as f:
        data = f.read().splitlines()
    predictions = []
    for masked_sent in tqdm(data, desc="Probing"):
        result_list = []
        for num in num_list:
            sent = masked_sent.replace("<mask>", num)
            input_ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)
            if cuda:
                input_ids = input_ids.to('cuda')
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss, logits = outputs[:2]
                sentence_prob = 100-loss.item()
                result_list.append({'word': num, 'softmax': sentence_prob})


        # result_list = model.predict_mask(masked_sent, options=num_list, num_results=1000)
        result_list.sort(key=lambda x:x["softmax"], reverse=True)
        # cast float32 to float
        result_list = [{"word":x["word"].lower(), "score":float(x["softmax"])} for x in result_list]
        output_str = json.dumps(dict(probe=masked_sent, result_list=result_list))
        predictions.append(output_str)
 
    with open(output_filename, 'w') as f:
        print("Saving resutls to", f.name)
        f.write("\n".join(predictions) + "\n")