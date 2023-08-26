import json
import os
import numpy as np
import pandas as pd

_VERSION = "1.0.0"

# Load the JSONL file
with open("datasets/filtered_dataset.jsonl", "r") as jsonl_file:
    dataset = [json.loads(line) for line in jsonl_file]

dataset = pd.DataFrame(dataset)  # Convert to DataFrame

print(f"Loaded {len(dataset)} examples.")

_JOURNALS = {
    "nature communications": "NC",
    "animals : an open access journal from mdpi": "A",
    "nihr journals library": "NIHR",
    "plos genetics": "PLGEN",
    "plos pathogens": "PLPAT",
    "plos computational biology": "PLCB",
    "plos neglected tropical diseases": "PLNTD",
    "biology": "B",
    "insects": "I",
    "elife": "EL",
    "plos biology": "PLB",
    "communications biology": "CB",
    "scientific data": "SD",
    "mbio": "MBIO",
    "cancers": "C",
    "others": "OTHER"
}

def save_examples(examples, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{os.path.basename(output_dir)}.jsonl"), "a") as jsonl_file:
        for example in examples:
            jsonl_file.write(json.dumps(example) + "\n")

journal_groups = dataset.groupby("journal")

others = []
for journal, examples in journal_groups:
    if journal in _JOURNALS:
        train, validate, test = np.split(examples.sample(frac=1, random_state=42), [int(.8*len(examples)), int(.90*len(examples))])
        save_examples(train.to_dict(orient="records"), f"sci_lay/data/{_VERSION}/train/{_JOURNALS[journal]}")
        save_examples(validate.to_dict(orient="records"), f"sci_lay/data/{_VERSION}/validation/{_JOURNALS[journal]}")
        save_examples(test.to_dict(orient="records"), f"sci_lay/data/{_VERSION}/test/{_JOURNALS[journal]}")
        print(f"Journal '{journal}' correctly splitted into train/validation/test.")
    else:
        others.append(examples)

other_examples = pd.concat(others, ignore_index=True)
train, validate, test = np.split(other_examples.sample(frac=1, random_state=42), [int(.8*len(other_examples)), int(.90*len(other_examples))])
save_examples(train.to_dict(orient="records"), f"sci_lay/data/{_VERSION}/train/OTHER")
save_examples(validate.to_dict(orient="records"), f"sci_lay/data/{_VERSION}/validation/OTHER")
save_examples(test.to_dict(orient="records"), f"sci_lay/data/{_VERSION}/test/OTHER")
print(f"Journals 'others' correctly splitted into train/validation/test.")
