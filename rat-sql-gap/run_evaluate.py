#!/usr/bin/env python

import _jsonnet
import json
import argparse
import collections
import attr
from seq2struct.commands import preprocess, train, infer, eval
from preprocess.preprocess_sparc import process_sparc_dev
import crash_on_ipy

from nltk import data
import os
data.path.append("nltk_data")

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()

@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()

@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib(default=None)
    use_heuristic = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)

@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="preprocess/train/eval")
    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
    args = parser.parse_args()
    
    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = json.dumps(exp_config["model_config_args"])
    else:
        model_config_args = None
    
    if args.mode == "preprocess":
        process_sparc_dev()
        preprocess_config = PreprocessConfig(model_config_file, model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        raise Exception('Cannot run training in evaluation script')
    elif args.mode == "eval":
        if not os.path.exists('predict'):
            os.makedirs('predict', exist_ok=True)
        infer_output_path = "predict/eval.infer"
        infer_config = InferConfig(
            model_config_file,
            model_config_args,
            exp_config["logdir"],
            exp_config["eval_section"],
            exp_config["eval_beam_size"],
            infer_output_path,
            step=None,
            use_heuristic=exp_config["eval_use_heuristic"]
        )
        infer.main(infer_config)

        eval_output_path = "predict/eval.eval"
        eval_config = EvalConfig(
            model_config_file,
            model_config_args,
            exp_config["logdir"],
            exp_config["eval_section"],
            infer_output_path,
            eval_output_path
        )
        eval.main(eval_config)

        res_json = json.load(open(eval_output_path))
        print(res_json['total_scores']['all']['exact'])

        post_process_predict()


def post_process_predict():
    n_turn_list = []
    raw_data = json.load(open('raw_data/sparc/dev.json'))
    for diag in raw_data:
        n_turns = len(diag['interaction'])
        n_turn_list.append(n_turns)

    pred_sql_list = []
    for line in open('predict/eval.infer'):
        try:
            pred = json.loads(line)['beams'][0]
            pred_sql = pred['inferred_code']
        except:
            pred_sql = 'select * from singer'
        pred_sql_list.append(pred_sql + '\n')
        n_turn_list[0] -= 1
        if n_turn_list[0] == 0:
            pred_sql_list.append('\n')
            del n_turn_list[0]

    with open('predict.txt', 'w') as fw:
        fw.writelines(pred_sql_list)


if __name__ == "__main__":
    main()