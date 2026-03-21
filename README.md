# Lab 3 - Setup a project  
CNN training on TinyImageNet with a cleaner project structure:  

- **models/**  
  → saved trained models  

- **datasets/**  
  → dataset files --> it's huge so it stays empty  

- **dataloaders/**  
  → dataset + dataloaders classes  

- **architectures/**  
  → PyTorch NN class  

- **utils/**  
  → helper functions (visualization, scripts, etc.)  

- **train.py**  
  → training logic  

- **eval.py**  
  → evaluation logic  

- **requirements.txt**  
  → pip packages (for reproducibility) 


lab-3/
│
├── checkpoints/          # saved model weights
├── data/                 # actual dataset files (NOT code)
│   └── tiny-imagenet/
│       └── tiny-imagenet-200/
│
├── dataset/              # dataset / dataloader code
│   └── dataset_and_dataloader.py
│
├── models/               # model architectures
│   └── NN_class.py
│
├── utils/                # helper functions
│   └── checkpoint_utils.py
│
├── .gitignore
├── README.md
├── eval.py               # evaluation logic
├── requirements.txt
└── train.py              # training logic

