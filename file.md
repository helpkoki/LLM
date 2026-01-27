my_llm/
│
├── data/
│   ├── raw/                # Original text datasets
│   ├── cleaned/            # Cleaned / tokenized text
│   ├── splits/             # train.txt, val.txt, test.txt
│   └── scripts/
│       └── preprocess.py
│
├── tokenizer/
│   ├── bpe.py
│   ├── sentencepiece.py
│   └── vocab.json
│
├── model/
│   ├── layers/
│   │   ├── attention.py
│   │   ├── feedforward.py
│   │   └── embeddings.py
│   │
│   ├── transformer.py
│   ├── config.py
│   └── weights/
│       └── checkpoint.npz
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── scheduler.py
│   └── callbacks.py
│
├── inference/
│   ├── generate.py
│   └── chat.py
│
├── utils/
│   ├── logger.py
│   ├── metrics.py
│   ├── helpers.py
│   └── seed.py
│
├── experiments/
│   ├── exp_01/
│   ├── exp_02/
│   └── notes.md
│
├── configs/
│   ├── base.yaml
│   ├── small.yaml
│   └── large.yaml
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_training.py
│
├── requirements.txt
├── README.md
└── main.py
