import math
import os
import numpy as np
from codecarbon import EmissionsTracker
from rank_bm25 import BM25Okapi


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

            """
            if np.argmax(bm25_scores) != i:
                answer = "Please provide the correct context."
            """
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
    return carburacy_train, carburacy_test, carburacy


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

