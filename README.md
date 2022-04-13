# RAT-SQL-TC
Code for our paper "Pay More Attention to History: A Context Modeling Strategy for Conversational Text-to-SQL"

Please refer to [gap-text2sql rep](https://github.com/awslabs/gap-text2sql) for environment and requirements.

### Preprocess dataset
```bash
python run.py preprocess experiments/sparc-tcs-configs/gap-run.jsonnet
```

## Inference
```bash
python run.py eval experiments/sparc-tcs-configs/gap-run.jsonnet
```

## Training
```bash
python run.py train experiments/sparc-tcs-configs/gap-run.jsonnet
```