# NumerSense: Probing Numerical Commonsense Knowledge of BERTs


Project website: https://inklab.usc.edu/NumerSense/

Code & Data for EMNLP 2020 paper:

```bibtex
@inproceedings{lin2020numersense,
  title={Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-trained Language Models},
  author={Bill Yuchen Lin and Seyeon Lee and Rahul Khanna and Xiang Ren}, 
  booktitle={Proceedings of EMNLP},
  year={2020},
  note={to appear}
}
```

## Installation 

```bash
conda create -n numersense python=3.7
conda activate numersense
# install torch seperately at https://pytorch.org/get-started/locally/ if needed
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch -n numersense
pip install transformers==3.3.1
# pip install happytransformer -U
pip install --editable happy-transformer
pip install tensorboardX

# Optional:
# Install apex following https://github.com/NVIDIA/apex#linux
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Probing Experiments 

For masked language models:
```bash
# folder for your personal results
mkdir pred_results

python src/mlm_predict.py bert-base \
        data/test.core.masked.txt \
        pred_results/bert-base.test.core.output.jsonl

python src/mlm_predict.py bert-base \
        data/test.all.masked.txt \
        pred_results/bert-base.test.all.output.jsonl
```

Note that `bert-base` can be replaced by any model name in `[bert-base, bert-large, roberta-base, roberta-large]`.

For left-to-right language models:
```bash
python src/gpt_predict.py gpt \
        data/test.core.masked.txt \
        pred_results/gpt.test.core.output.jsonl 
```

### Fine-tune a MLM model 
```bash
mkdir saved_models
CUDA_VISIBLE_DEVICES=0 python src/finetune_mlm.py \
  --output_dir=saved_models/finetuned_bert_large --overwrite_output_dir \
  --model_type=bert \
  --model_name_or_path=bert-large-uncased \
  --do_train \
  --train_data_file=data/gkb_best_filtered.txt  \
  --do_eval \
  --eval_data_file=data/wiki_complete.txt \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --block_size 64 \
  --logging_steps 100 \
  --num_train_epochs 3 \
  --line_by_line --mlm 
```

```bash 
python src/mlm_predict.py \
        reload_bert:saved_models/finetuned_bert_large \
        data/test.core.masked.txt \
        pred_results/test.core.output.jsonl
```

## Evaluation on Validation Set

Check out `data/validation.masked.tsv`. We realease 200 annotated examples (132 from the `core` split and 68 from the `all` split) for method development so that users can better test their method frquently without submitting the prediction for the test set. **Note that these 200 examples should NOT be used for any training.** Also, they are still part of the the test data.

## Evaluation on Test Set

To evaluate your model's ability on NumerSense's official test sets,
please submit a prediction file to *yuchen.lin@usc.edu*, which should contain a json line for each probe example. And a json line should follow the format in the below code snippet. You can also check the example, `results/bert-base.test.core.output.jsonl` , which is the predictions of BERT-base on core set.
The `score` key is optional.
When submitting your predictions, please submit both `core` and `all` results, and inform us whether you have used the training data for fine-tuning. Thanks!
The evaluation script we will use is `src/evaluator.py`.
 ```json
{
  "probe": "a bird has <mask> legs.",
  "result_list": [
    {
      "word": "four",
      "score": 0.23623309
    },
    {
      "word": "two",
      "score": 0.21001829
    },
    {
      "word": "three",
      "score": 0.1258428
    },
    {
      "word": "no",
      "score": 0.0688955
    },
    {
      "word": "six",
      "score": 0.0639159
    },
    {
      "word": "five",
      "score": 0.061465383
    },
    {
      "word": "eight",
      "score": 0.038915534
    },
    {
      "word": "seven",
      "score": 0.014524153
    },
    {
      "word": "ten",
      "score": 0.010337788
    },
    {
      "word": "nine",
      "score": 0.005654324
    },
    {
      "word": "one",
      "score": 1.3131318E-4
    },
    {
      "word": "zero",
      "score": 1.10984496E-4
    }
  ]
}
 ```
