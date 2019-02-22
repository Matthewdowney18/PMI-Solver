import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/IR_output_0", required=False)
    parser.add_argument("dataset_filename", required=False, default="fiction_reformatted_1/dev_1.json")

    args = parser.parse_args()

    with open(args.dataset_filename) as f:
        dataset=json.load(f)

    results = {}
    for entry in tqdm(dataset['data']):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            #make pmi object here

            for qa in paragraph["qas"]:
                #use pmi object for input and results
                result = 2
                results[qa["id"]] = result

    metrics = evaluate_results(results, dataset["data"])
    with open("output_0.json", "w") as f:
        json.dump(metrics, f, indent=4)

def evaluate_results(outputs, dataset):
    metrics = {}
    predictions = list()
    targets = list()
    for entry in dataset:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["is_impossible"] == True:
                    targets.append(0)
                else:
                    targets.append(1)
                predictions.append(outputs[qa["id"]]["answer"] + 1)
                outputs[qa["id"]]["target"] = targets[-1] - 1

    metrics["unaswerable"] = get_metrics_for_label(targets, predictions, 0)
    metrics["correct_answer"] = get_metrics_for_label(targets, predictions,
                                                      1)

    #metrics["accuracy"] = accuracy_score(targets, predictions)

    metrics["outputs"] = outputs
    return metrics

def get_metrics_for_label(targets, predictions, label):
    metrics = {}
    true_positives = [1 for target, prediction in zip(targets, predictions)
                      if
                      (target == label and prediction == label)]
    false_positives = [1 for target, prediction in zip(targets, predictions)
                       if
                       (target != label and prediction == label)]
    false_negatives = [1 for target, prediction in zip(targets, predictions)
                       if
                       (target == label and prediction != label)]

    true_positives = sum(true_positives)
    false_positives = sum(false_positives)
    false_negatives = sum(false_negatives)

    metrics["precision"] = true_positives / (
                true_positives + false_positives)
    metrics["recall"] = true_positives / (true_positives + false_negatives)
    metrics["F1"] = 2 * metrics["precision"] * metrics["recall"] / (
                metrics["precision"] + metrics["recall"])

    return metrics


if __name__ == "__main__":
    main()
