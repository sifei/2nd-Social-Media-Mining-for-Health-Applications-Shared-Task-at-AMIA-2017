# Deep model for 2nd Social Media Mining for Health Applications Shared Task at AMIA 2017

Implementation of CNN and attention based CNN model.
 * By default, training, deveploment and test set on ``data_sample`` directory.

## Required Packages
- Python 2.7
- numpy
- theano
- sklearn
- nltk

## Usage

Please note that this model takes as input the path of the data folder and not paths to each individual file. A data folder is expected to have the following files: `train_ids.txt`, `dev_ids.txt`, `test_ids.txt`, and `dataset.txt`. Please see the example_dataset directory for an example of the format. 

- More info about the dataset format can be found [here](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task1/data_sample).



### Training and Evaluating

```
bash run_loop.sh
```

```
usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--num_models NUM_MODELS]
                [--lstm_hidden_state LSTM_HIDDEN_STATE] [--word_vectors WORD_VECTORS]
                [--checkpoint_dir CHECKPOINT_DIR]
                [--checkpoint_name CHECKPOINT_NAME] [--hidden_state HIDDEN_STATE]
                [--learn_embeddings LEARN_EMBEDDINGS] [--min_df MIN_DF] [--lr LR]
                [--penalty PENALTY] [--p_penalty P_PENALTY] [--dropout DROPOUT]
                [--lstm_dropout LSTM_DROPOUT] [--lr_decay LR_DECAY] [--minibatch_size MINIBATCH_SIZE]
                [--val_minibatch_size VAL_MINIBATCH_SIZE] [--model_type MODEL_TYPE] [--train_data_X TRAIN_DATA_X]
                [--train_data_Y TRAIN_DATA_Y] [--val_data_X VAL_DATA_X] [--val_data_Y VAL_DATA_Y]
                [--seed SEED] [--grad_clip GRAD_CLIP] [--cnn_conv_size CNN_CONV_SIZE]
                [--num_feat_maps NUM_FEAT_MAPS] [--num_att NUM_ATT] 

Train and evaluate MODEL on a given dataset

optional arguments:
  -h, --help            show this help message and exit
  --train_data_X TRAIN_DATA_X   
                        path to the train dataset
  --train_data_X TRAIN_DATA_Y   
                        path to the train dataset labels
  --val_data_X VAL_DATA_X   
                        path to the dev/test dataset
  --val_data_X VAL_DATA_Y   
                        path to the dev/test dataset labels
  --batch-size BATCH_SIZE
                        number of instances in a minibatch
  --num-epoch NUM_EPOCH
                        number of passes over the training set
  --learning-rate LEARNING_RATE
                        learning rate, default depends on optimizer
  --word_vectors WORD_VECTORS
                        word vectors filepath
  --embedding-factor EMBEDDING_FACTOR
                        learning rate multiplier for embeddings
  --lr_decay LR_DECAY_RATE    exponential decay for learning rate
  --seed SEED           seed for training

```

## Acknowledgements

Please consider citing the following paper(s) if you use this software in your work:

> Sifei Han, Tung Tran, Anthony Rios, and Ramakanth Kavuluru. "Team UKNLP: Detecting ADRs, Classifying Medication Intake Messages, and Normalizing ADR Mentions on Twitter" In Proceedings of the 2nd Social Media Mining for Health Research and Applications Workshop
co-located with the American Medical Informatics Association Annual Symposium (AMIA 2017), vol-1996, pp. 49-53. 2017.

```
@inproceedings{kavuluru2017extracting,
  title={Team UKNLP: Detecting ADRs, Classifying Medication Intake Messages, and Normalizing ADR Mentions on Twitter},
  author={Han, Sifei and Tran, Tung and Rios, Anthony and Kavuluru, Ramakanth},
  booktitle={Social Media Mining for Health Research and Applications},
  pages={49--53},
  year={2017},
  organization={AMIA}
}
```


You can get the embeddings here:
https://nlp.stanford.edu/projects/glove/

## Author

> Sifei Han 
> sehan2 **[at]** g.uky.edu
> Anthony Rios
> anthonymrios **[at]** gmail.com


