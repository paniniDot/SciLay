import argparse
import json
import os
import time
import torch
import math 

import numpy as np
import rouge
from BARTScore.bart_score import BARTScorer
from codecarbon import EmissionsTracker
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate
from utils import compute_sim

def generate_summary(document, guidelines, tokenizer, ll_model, max_source_length, device):
    instruction = f"Summarize the following document according to these guidelines: {guidelines}. Document: {document}"
    inputs = tokenizer(instruction, return_tensors="pt", max_length=max_source_length, truncation=True).to(device)

    new_token = torch.tensor([[tokenizer.additional_special_tokens_ids[0]]]).to(device)
  
    input_ids_tensor = torch.cat((inputs['input_ids'], new_token), dim=1)

    # Passa un dizionario con la chiave 'input_ids' al metodo generate
    summary_ids = ll_model.generate(input_ids=input_ids_tensor, max_length=512)
    dec_out = tokenizer.batch_decode(summary_ids, skip_special_tokens=False)
    return dec_out[0].split('[END]')[1]

def compute_metrics(predictions, references, device):
    global_rouge_scorer = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                      max_n=4,
                                      limit_length=True,
                                      length_limit=100,
                                      length_limit_type='words',
                                      apply_avg=True,
                                      apply_best=False,
                                      alpha=0.5,  # Default F1_score
                                      weight_factor=1.2,
                                      stemming=True)   
    metric_bertscore = evaluate.load("bertscore")
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    sim_model = SentenceTransformer('sentence-transformers/roberta-large-nli-stsb-mean-tokens').to(device)
    
    result = {}

    # ROUGE scores
    rouge_scores = global_rouge_scorer.get_scores(hypothesis=predictions, references=references)
    result["rouge1"] = round(100 * rouge_scores["rouge-1"]["f"], 2)
    result["rouge2"] = round(100 * rouge_scores["rouge-2"]["f"], 2)
    result["rougeL"] = round(100 * rouge_scores["rouge-l"]["f"], 2)

    # BLEU scores
    tokenized_predictions = [prediction.split(" ") for prediction in predictions]
    tokenized_references = [[reference.split(" ")] for reference in references]
    result["bleu1"] = round(100 * corpus_bleu(tokenized_references, tokenized_predictions, weights=(1, 0, 0, 0)), 2)
    result["bleu2"] = round(100 * corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.5, 0.5, 0, 0)), 2)
    result["bleu3"] = round(100 * corpus_bleu(tokenized_references, tokenized_predictions, weights=(1/3, 1/3, 1/3, 0)), 2)
    result["bleu4"] = round(100 * corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.25, 0.25, 0.25, 0.25)), 2)

    result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
        (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

    # BERTScore
    result_bs = metric_bertscore.compute(predictions=predictions, references=references, lang="en", idf=True, rescale_with_baseline=True)
    result["bertscore"] = round(100 * np.mean(result_bs["f1"]), 2)

    # BARTScore
    bartr_scores = bart_scorer.score(predictions, references)
    bartp_scores = bart_scorer.score(references, predictions)
    result["bart_score_R"] = round(np.mean(bartr_scores), 3)
    result["bart_score_P"] = round(np.mean(bartp_scores), 3)
    result["bart_score_F"] = round(np.mean([(pscore + rscore) / 2 for pscore, rscore in zip(bartp_scores, bartr_scores)]), 3)

    # Mean cosine similarity
    result["mean_cos_sim"] = compute_sim(sim_model, references, predictions)

    # Mean generation length
    result["gen_len"] = np.mean([len(prediction.split()) for prediction in predictions])

    return result


def get_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
    carburacy_train = None
    if emission_train is not None:
        carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
        carburacy_train = round(100 * carburacy_train, 2)
    carburacy_test = None
    if emission_test is not None:
        carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
        carburacy_test = round(100 * carburacy_test, 2)
    carburacy = None
    if carburacy_train is not None and carburacy_test is not None:
        carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
        carburacy = round(100 * carburacy, 2)
    return carburacy_train, carburacy_test, carburacy


def main():

    output_prediction_path = os.path.join(args.output_dir, f"{args.model.split('/')[1]}_{args.dataset_subset}_{args.max_source_length}_summarization")
    if not os.path.exists(output_prediction_path):
      os.makedirs(output_prediction_path)

    raw_datasets = load_dataset(args.dataset_name, args.dataset_subset)
    train_dataset = raw_datasets[args.split]
    references = train_dataset["plain_text"]  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"IL DEVICE IN USO È QUESTO {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ll_model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="../llms", load_in_4bit=True, trust_remote_code=True, device_map="auto")

    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': [" [END]"]})
    ll_model.resize_token_embeddings(len(tokenizer))

    tracker = EmissionsTracker(output_dir=output_prediction_path)
    tracker.start()

    start_time = time.time()

    predictions = []
    with torch.no_grad():
        for document in tqdm(train_dataset["full_text"]):
            summary = generate_summary(document, args.guidelines, tokenizer, ll_model, args.max_source_length, device)
            predictions.append(summary)
    
    end_time = time.time()

    emissions = tracker.stop()
    elapsed_time = float(f"{(end_time - start_time):.2f}")
    results = compute_metrics(predictions, references, device)

    results["emissions"] = emissions
    results["runtime"] = elapsed_time
    _, results["carburacy"], _ = get_carburacy(results["R"],
                                            None, 
                                            emissions/len(train_dataset))
 


    with open(os.path.join(output_prediction_path, f"generated_{args.split}_set.txt"), "w") as writer:
        writer.write("\n".join(predictions))

    with open(os.path.join(output_prediction_path, f"results_{args.split}.json"), "w") as writer:
        json.dump(results, writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--dataset_subset", default=None, type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--max_source_length", default=None, type=int)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--guidelines", required=True, type=str, help="Guidelines for the summarization task.")
    args = parser.parse_args()

    main()