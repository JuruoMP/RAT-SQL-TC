import argparse
import json
import os

import _jsonnet
import tqdm

from seq2struct import datasets
from seq2struct import models
from seq2struct.utils import registry
from seq2struct.utils import vocab

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        self.eval_mode = True
        if self.eval_mode:
            self.model_preproc.enc_preproc.load('static/enc')
            self.model_preproc.dec_preproc.load('static/dec')

    def preprocess(self):
        self.model_preproc.clear_items()
        for section in self.config['data']:
            # if section=="train":
            #     continue
            data = registry.construct('dataset', self.config['data'][section])
            for item in tqdm.tqdm(data, desc=section, dynamic_ncols=True):
                if True:
                    to_add, validation_info = self.model_preproc.validate_item(item, section)
                    if to_add:
                        self.model_preproc.add_item(item, section, validation_info)
                else:# except Exception as e:
                    print(e)
                    print("======== Error parsing: {}".format(" ".join(item.text)))
        self.model_preproc.save()

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args

def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    preprocessor = Preprocessor(config)
    preprocessor.preprocess()

if __name__ == '__main__':
    args = add_parser()
    main(args)
