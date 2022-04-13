This folder contains the SParC training and dev data sets for our ACL 2019 paper "SParC: Cross-Domain Semantic Parsing in Context".

It contains the following files:

- train.json: the SParC training set following the same DB split as the Spider dataset does.
- dev.json: the SParC dev set following the same DB split as the Spider dataset does.
- tables.json: the file including all DB schema info such as table and column names.
- dev_gold.txt: the file as a gold input for the SParC evaluation
- database/: each folder for each database which contains the [db_name].sqlite and schema.sql files.
- README.txt

For the format of each JSON file, please refer to our Github page: https://github.com/taoyds/sparc.
For the details of the SParC, Yale & Salesforce Semantic Parsing and Text-to-SQL in Context Challenge, please refer to our task page: https://yale-lily.github.io/sparc


If you use the dataset, please cite the following papers:

@InProceedings{Yu&al.19,
  title     = {SParC: Cross-Domain Semantic Parsing in Context},
  author    = {Tao Yu and Rui Zhang and Michihiro Yasunaga and Yi Chern Tan and Xi Victoria Lin and Suyi Li and Heyang Er, Irene Li and Bo Pang and Tao Chen and Emily Ji and Shreya Dixit and David Proctor and Sungrok Shim and Jonathan Kraft, Vincent Zhang and Caiming Xiong and Richard Socher and Dragomir Radev},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year      = {2019},
  address   = {Florence, Italy},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{Yu&al.18c,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev}
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  address   = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  year      = 2018
}


Reference links

SParC task link: https://yale-lily.github.io/sparc
SParC Github page: https://github.com/taoyds/sparc
Spider task link: https://yale-lily.github.io/spider
Spider Github page: https://github.com/taoyds/spider