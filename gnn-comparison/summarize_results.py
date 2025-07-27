import os
import json
import csv
import argparse



def get_args_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument('RESULTS_DIR',
                        help='path of the results directory')
    parser.add_argument('--out_file', dest='output_file',
                        default='all', help='dataset name [Default: \'all\']')

    return vars(parser.parse_args())


def summarize_all_results_to_csv(results_root, output_file):
    fieldnames = [
        "model", "dataset", "rewiring_strategy", "rewire_all_layers",
        "num_layers", "lr", "embedding_dim", "hidden_dim", "batch_size",
        "avg_TR_score", "std_TR_score", "avg_TS_score", "std_TS_score"
    ]

    rows = []

    for experiment_name in sorted(os.listdir(results_root)):
        experiment_path = os.path.join(results_root, experiment_name, "10_NESTED_CV")
        if not os.path.isdir(experiment_path):
            continue

      
        row = dict.fromkeys(fieldnames, None)

        #getting experiment name
        parts = experiment_name.split("_")
        if len(parts) >= 2:
            row["model"] = parts[0]
            row["dataset"] = parts[1]
        if len(parts) > 2:
            row["rewiring_strategy"] = "_".join(parts[2:])

        found_config = False

        # assessment scores
        assessment_path = os.path.join(experiment_path, "assessment_results.json")
        if os.path.exists(assessment_path):
            with open(assessment_path) as f:
                assessment = json.load(f)
                row["avg_TR_score"] = round(assessment.get("avg_TR_score", 0), 2)
                row["std_TR_score"] = round(assessment.get("std_TR_score", 0), 2)
                row["avg_TS_score"] = round(assessment.get("avg_TS_score", 0), 2)
                row["std_TS_score"] = round(assessment.get("std_TS_score", 0), 2)

        #adding config: 
        config_path = os.path.join(experiment_path, "OUTER_FOLD_9", "outer_results.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                fold = json.load(f)
                if "best_config" in fold and "config" in fold["best_config"]:
                    config = fold["best_config"]["config"]
                    row["num_layers"] = config.get("num_layers")
                    row["lr"] = config.get("learning_rate")
                    row["batch_size"] = config.get("batch_size")
                    row["hidden_dim"] = config.get("hidden_units")
                    row["embedding_dim"] = config.get("dim_embedding")
                    row["rewire_all_layers"] = config.get("rewire_for_all_layers")
                    found_config = True

        if found_config:
            rows.append(row)
        else:
            print(f"Skipping {experiment_name}: no usable config found.")


    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f" file saved in {output_file}")




if __name__ == "__main__":
    
    args_dict = get_args_dict()
    print(args_dict)

    results_path = args_dict['RESULTS_DIR']
    output_file = args_dict['output_file']
    summarize_all_results_to_csv(results_path, output_file)
#python summarize_results.py /teamspace/studios/this_studio/gnn-comparison/RESULTS --out_file summary.csv