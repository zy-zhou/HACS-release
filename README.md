## Requirements
* Python 3.7
* PyTorch 1.4
* torchtext 0.4
* CUDA & CuDNN to run on GPU (recommended)
* javalang 0.13
* pyastyle 1.1.5
* tqdm 4.47
* numpy 1.18.1
* NLTK 3.5
* py-rouge 1.1

## Data
The original data we used is from https://github.com/xing-hu/TL-CodeSum and https://github.com/xing-hu/EMSE-DeepCom. This repository only contains a tiny dataset to verify whether the scripts work. Be aware that this dataset is not big enough to properly train a seq2seq model.  

If you want to try some new data (in Java), simply reference and replace **raw_data.json** in the **data** folder. The code to shuffle and remove the duplicate samples is not included so you have to manually do this if required.

## Parsing & Preprocessing
* First parse the codes into ASTs and then split them into statement subtrees. Codes that cannot be parsed or less than 2 statements (e.g., empty body) will be marked for discard.  
  `python Tree.py`
* Filter & tokenize the codes and comments, and then split the codes by statement. A formatter (Artistic Style) is used here. Comments with less than 1 token after tokenization will be marked for discard. The remaining samples will be split into train/valid/test sets. Vocabularies will also be generated during this step.  
  `python Data.py`

## Build & Train the model
* The **Train** script will pretrain a pair of seq2seq models on token/node sequences and then use their encoders to initialize HACS-token/AST. After that, HACS-token/AST will be trained.  
  `python Train.py`

## Test & Evaluate the model
* Print the scores on test set and save the predictions.  
  `python Main.py`

## Change the Parameters
All parameters are defined in the beginning of the .py files. We will make the scripts easier to use in the future (e.g., by using argparse).
