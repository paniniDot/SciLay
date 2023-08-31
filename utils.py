import math
import os
import torch
import rouge
import evaluate
import numpy as np
from statistics import mean
from collections import Counter
from sentence_transformers import util
from codecarbon import EmissionsTracker
from rank_bm25 import BM25Okapi
from BARTScore.bart_score import BARTScorer
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu



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


def compute_nli(model_nli, references, predictions):
    scores_nli = model_nli.predict([[l, p] for l, p in zip(references, predictions)])
    
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores_nli.argmax(1)]
    label_counts = Counter(labels)
    total_labels = len(labels)

    label_proportions = {label: round(100 * count / total_labels, 2) for label, count in label_counts.items()}

    return label_proportions

def compute_sim(sim_model, references, predictions):

    cos_sim_list = []
    for ref, pred in zip(references, predictions):
        embedding_ref = sim_model.encode(ref, convert_to_tensor=True)
        embedding_pred = sim_model.encode(pred, convert_to_tensor=True)

        cos_sim_list.append(util.pytorch_cos_sim(embedding_ref, embedding_pred).item())
        
    return round(100 * mean(cos_sim_list), 2)


def construct_bm25(raw_dataset, data_args):
    
    corpus_full = []
    tokenized_corpus_full = []
    for split in ["train", "validation", "test"]:
        tokenized_corpus_split = [doc.split(" ") for doc in raw_dataset[split][data_args.input_column]]
        tokenized_corpus_full.extend(tokenized_corpus_split)
        corpus_full.extend(raw_dataset[split][data_args.input_column])
    
    bm25 = BM25Okapi(tokenized_corpus_full)

    return bm25, corpus_full


def construct_inp_target(examples, data_args, perform_retrieval, bm25, corpus_full):

    inputs, targets = [], []
    for i, (document, qa_pair) in enumerate(zip(examples[data_args.input_column], examples[data_args.target_column])):
        question = qa_pair.split("Answer")[0].strip()
        answer = qa_pair.split("Answer: ")[1].strip()

        if perform_retrieval:
            tokenized_question = question.split(" ")
            bm25_scores = bm25.get_scores(tokenized_question)
            query = question + " Context: " + corpus_full[np.argmax(bm25_scores)]

            if data_args.altered_answ and corpus_full[np.argmax(bm25_scores)] != document:
                answer = "Please provide the correct context."
                
        else:
            query = question + " Document: " + document
        
        inputs.append(query)
        targets.append(answer)
    
    return inputs, targets


def get_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
    carburacy_train = None
    if emission_train is not None:
        carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
    carburacy_test = None
    if emission_test is not None:
        carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
    carburacy = None
    if carburacy_train is not None and carburacy_test is not None:
        carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
    return round(100 * carburacy_train, 2), round(100 * carburacy_test, 2), round(100 * carburacy, 2)


def predict(trainer, predict_dataset, max_predict_samples, training_args, tokenizer, train_emissions, split):
    test_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
    test_tracker.start()
    predict_results = trainer.predict(predict_dataset, metric_key_prefix=split)
    test_emissions = test_tracker.stop()

    metrics = predict_results.metrics

    metrics[f"{split}_samples"] = min(max_predict_samples, len(predict_dataset))
    metrics[f"{split}_emissions"] = test_emissions

    if training_args.do_train:
        train_carburacy, predict_carburacy, carburacy = get_carburacy(metrics[f"{split}_R"], 
                                                                    train_emissions, test_emissions/len(predict_dataset))
        metrics["train_carburacy"] = train_carburacy
        metrics[f"{split}_carburacy"] = predict_carburacy
        metrics["carburacy"] = carburacy

    trainer.log_metrics(split, metrics)
    trainer.save_metrics(split, metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, f"generated_{split}_set.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))


def compute_metrics(references, predictions, sim_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_bertscore = evaluate.load("bertscore")
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')

    rouge_scores = global_rouge_scorer.get_scores(hypothesis=predictions, references=references)
    result = {"rouge1": round(100 * rouge_scores["rouge-1"]["f"], 2),
                "rouge2": round(100 * rouge_scores["rouge-2"]["f"], 2),
                "rougeL": round(100 * rouge_scores["rouge-l"]["f"], 2),
            }
    
    # Compute BLEU scores
    tokenized_predictions = [prediction.split(" ") for prediction in predictions]
    tokenized_labels = [[ref.split(" ")] for ref in references]

    result["bleu1"] = round(100 * corpus_bleu(tokenized_labels, tokenized_predictions, weights=(1, 0, 0, 0)), 2)
    result["bleu2"] = round(100 * corpus_bleu(tokenized_labels, tokenized_predictions, weights=(1/2, 1/2, 0, 0)), 2)
    result["bleu3"] = round(100 * corpus_bleu(tokenized_labels, tokenized_predictions, weights=(1/3, 1/3, 1/3, 0)), 2)
    result["bleu4"] = round(100 * corpus_bleu(tokenized_labels, tokenized_predictions, weights=(1/4, 1/4, 1/4, 1/4)), 2)
    

    result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
        (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

    result_bs = metric_bertscore.compute(predictions=predictions, references=references, lang="en",
                                            idf=True, rescale_with_baseline=True,
                                            model_type="bert-base-uncased")
    result["bertscore"] = round(sum(result_bs["f1"]) / len(result_bs["f1"]) * 100, 2)

    bartr_scores = bart_scorer.score(predictions, references)
    bartp_scores = bart_scorer.score(references, predictions)

    bart_score_R = mean(bartr_scores)
    bart_score_P = mean(bartp_scores)
    bart_score_F = mean([mean([pscore, rscore]) for pscore, rscore in zip(bartp_scores, bartr_scores)])
    result["bart_score_R"] = round(bart_score_R, 3)
    result["bart_score_P"] = round(bart_score_P, 3)
    result["bart_score_F"] = round(bart_score_F, 3)

    result["mean_cos_sim"] = compute_sim(sim_model, references, predictions)
    
    return result