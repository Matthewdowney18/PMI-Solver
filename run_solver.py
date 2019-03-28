import json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from PMI import PMI
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="fiction_3_28/result_0", required=False)
    parser.add_argument("--dataset_filename", default="fiction_3_28/data_0/test.json", required=False)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #raise ValueError("Output directory ({})"
        #                 " already exists and is not empty.".format(
        #    args.output_dir))
    #else:
        os.makedirs(args.output_dir)

    with open(args.dataset_filename) as f:
        dataset=json.load(f)
    total_accuracy = []
    metrics = {}
    all_results = {}
    question_data = []
    for question_type, data in tqdm(dataset['data'].items()):
        results = {}
        for entry in data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]

                solver = PMI(paragraph_text, 4, 10)

                for qa in paragraph["qas"]:

                    result = get_highest_scored_answer(qa, solver)
                    results[qa["id"]] = result

                    question_data.append((qa["id"], 1 if result["answer"] == result["label"] else 0))
        all_results[question_type] = results
        metrics[question_type] = evaluate_results(results)
        total_accuracy.append(metrics[question_type]["accuracy"])
        metrics["total_accuracy"] = np.mean(total_accuracy)
        with open(os.path.join(args.output_dir, "all_results"), "w") as f:
            json.dump(all_results, f, indent=4)
        with open(os.path.join(args.output_dir, "output.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        df = pd.DataFrame(question_data)
        df.to_csv(os.path.join(args.output_dir,"heatmap_data.csv"), index=False)

def get_highest_scored_answer(qa, search):

    question_text = qa["question"]

    is_impossible = qa["type"] == "Unanswerable"

    answer_texts = []
    for i in range(0, 4):
        answer_text = qa["answer_{}".format(i)]
        if answer_text["text"] != "not enough information":
            answer_texts.append(answer_text)
        elif answer_text["correct"] is True:
            label = -1

    for i, j in enumerate(answer_texts):
        if j["correct"] is True:
            label = i

    result = search.get_result(qa, answer_texts)
    result["label"] = label
    return result


def evaluate_results(outputs):
    metrics = {}

    targets = [output["label"] for output in outputs.values()]
    predictions = [output["answer"] for output in outputs.values()]

    metrics["accuracy"] = accuracy_score(targets, predictions)
    num_neg_predictions = len([1 for prediction in predictions if (prediction == -1)])
    metrics["unanswerable_predictions_ratio"] =  num_neg_predictions/len(predictions)
    return metrics

if __name__ == "__main__":
    main()
