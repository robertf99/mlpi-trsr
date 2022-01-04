Fork from https://github.com/fulifeng/Temporal_Relational_Stock_Ranking

## Changes
1. Migrated from TF1.x to compat.v1 API symbols via `tf_upgrade_v2`
1. Added training checkpoints under `./training_v2/.checkpoints`
2. Replaced Const Ops with Placeholder to reduce GraphDef (which saves values form Const Ops) size, so it can be loaded to Tensorboard via exported events (`FileWriter`) or `.pb` files.
3. Added summaries to tracking losses from training, validation and testing
## Setup
```
pipenv install --skip-lock -d
```
## Run

Use `training_v2` folder to run with Tensorflow 2.x.
```
python relation_rank_lstm.py -m NYSE -l 8 -u 32 -a 10 -e NYSE_rank_lstm_seq-8_unit-32_0.csv.npy
```
## Tensorboard
```
tensorboard --logdir ./training_v2/logs
```

## Notes
- `tf.compat.v1.Saver` used for training checkpoint instead of CheckingPoint due to current optimizer (AdamOptimizer) is not trackable
- The original implementation used `tf1.const` for `relation` and `rel_mask` ops, which is saved to GraphDef and too big to load to Tensorboard. To circumvent, we can extract GraphDef from exported model by stripping the `tensor_content` field of graph node and then manually upload to Tensorboard via following command. However it is still better to replace `tf1.const` to `tf1.placeholder`
  ```
  pipenv shell
  python -c "from utils import save_strip_pbtxt; save_strip_pbtxt('./model_outputs/NYSE/{#}')" 
  ```


For details, see below orginal Readme.md



# Orignal README.MD
Code for the Relational Stock Ranking (RSR) model and the Temporal Graph Convolution in our paper "Temporal Relational Ranking for Stock Prediction", [\[paper\]](https://arxiv.org/abs/1809.09441).

## Environment

Python 3.6 & Tensorflow > 1.3



## Data

All data, including Sequential Data, Industry Relation, and Wiki Relation, are under the [data](https://github.com/hennande/Temporal_Relational_Stock_Ranking/tree/master/data) folder.

### Sequential Data

Raw data: files under the [google_finance](https://github.com/hennande/Temporal_Relational_Stock_Ranking/tree/master/data/google_finance) folder are the historical (30 years) End-of-day data (i.e., open, high, low, close prices and trading volume) of more than 8,000 stocks traded in US stock market collected from Google Finance.

Processed data: [2013-01-01](https://github.com/hennande/Temporal_Relational_Stock_Ranking/tree/master/data/2013-01-01) is the dataset used to conducted experiments in our paper.

To get the relation data, run the following command:
```
tar zxvf relation.tar.gz
```

### Industry Relation

Under the sector_industry folder, there are row relation file and binary encoding file (.npy) storing the industry relations between stocks in NASDAQ and NYSE.

### Wiki Relation

Under the wikidata folder, there are row relation file and binary encoding file (.npy) storing the Wiki relations between stocks in NASDAQ and NYSE.

## Code

### Pre-processing

| Script | Function |
| :-----------: | :-----------: |
| eod.py | To generate features from raw End-of-day data |
| sector_industry.py | Generate binary encoding of industry relation |
| wikidata.py | Generate binary encoding of Wiki relation |

### Training
| Script | Function |
| :-----------: | :-----------: |
| rank_lstm.py | Train a model of Rank_LSTM |
| relation_rank_lstm.py | Train a model of Relational Stock Ranking |


## Run

To repeat the experiment, i.e., train a RSR model, downloaded the pretrained [sequential embedding](https://drive.google.com/file/d/1fyNCZ62pEItTQYEBzLwsZ9ehX_-Ai3qT/view?usp=sharing), and extract the file into the data folder.

### NASDAQ
```
python relation_rank_lstm.py -rn wikidata -l 16 -u 64 -a 0.1
```

### NYSE
```
python relation_rank_lstm.py -m NYSE -l 8 -u 32 -a 10 -e NYSE_rank_lstm_seq-8_unit-32_0.csv.npy
```

to enable gpu acceleration, add the flag of:
```
-g 1
```

## Cite

If you use the code, please kindly cite the following paper:
```
@article{feng2019temporal,
  title={Temporal relational ranking for stock prediction},
  author={Feng, Fuli and He, Xiangnan and Wang, Xiang and Luo, Cheng and Liu, Yiqun and Chua, Tat-Seng},
  journal={ACM Transactions on Information Systems (TOIS)},
  volume={37},
  number={2},
  pages={27},
  year={2019},
  publisher={ACM}
}
```

## Contact

fulifeng93@gmail.com
