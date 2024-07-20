import argparse
from transformers import AutoTokenizer
import nltk
import evaluate
import numpy as np
import json



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True,
                        help="Path to Llama2-70b-hf-chat checkpoint")
    parser.add_argument("--mlperf-accuracy-file", required=True,
                        help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-file", required=True,
                        help="path to processed openorca validation set")
    parser.add_argument("--verbose", action="store_true",
                        help="verbose messages")
    parser.add_argument("--dtype", default="int64",
                        help="dtype of the accuracy log", choices=["int32", "int64", "float"])
    args = parser.parse_args()
    return args


def get_groundtruth(processed_dataset_file):
    import pandas as pd
    data = pd.read_pickle(processed_dataset_file)
    ground_truths = data['output']
    return ground_truths

def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets



def main():

    args = get_args()
    dataset_path = args.dataset_file
    checkpoint_path = args.checkpoint_path
    metric = evaluate.load("rouge")
    nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,)


    targets = get_groundtruth(args.dataset_file)

    target_required = []
    preds_token_ids = []

    eval_dtype = np.int64
    if args.dtype == "int32":
        eval_dtype = np.int32
    elif args.dtype == "float":
        eval_dtype = np.float32

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    gen_tok_len = 0
    for pred in results:
        qsl_idx = pred['qsl_idx']
        if qsl_idx in seen:
            continue

        seen.add(qsl_idx)
        target = targets[qsl_idx]
        target_required.append(target)
        pred = np.frombuffer( bytes.fromhex(pred['data']), eval_dtype)
        if pred[0] > 32000 or pred[0] < 0:
            pred = [1, *pred[1:]]
        gen_tok_len += len(pred)
        preds_token_ids.append(pred)

    preds_decoded_text = tokenizer.batch_decode(
        preds_token_ids, skip_special_tokens=True)

    preds, targets = postprocess_text(preds_decoded_text, target_required)

    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False)
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    gen_num = len(preds)

    result = {**result,
              'gen_len': np.sum(prediction_lens),
              'gen_num': gen_num,
              'gen_tok_len': gen_tok_len,
              'tokens_per_sample': round(gen_tok_len / gen_num, 1)
              }

    print("\nResults\n")
    print(result)


if __name__ == "__main__":
    main()
