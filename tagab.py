import logging
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from flair.data import Corpus

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BytePairEmbeddings, \
    CharacterEmbeddings, FlairEmbeddings, BertEmbeddings, OneHotEmbeddings
from typing import List
from more_hot import MoreHotEmbeddings

# 1. get the corpus
import flair.datasets
from flair.training_utils import add_file_handler
from flair.models import SequenceTagger
from polish_benchmarks.helpers import get_embeddings
from tsv import TSVCorpus

parser = ArgumentParser(description='Train')

parser.add_argument('--hidden_size', default=256, type=int, help='size of embedding projection')
parser.add_argument('--downsample', default=1.0, type=float, help='downsample ratio')
parser.add_argument('--output_folder', default='dot1', help='output folder for log and model')
parser.add_argument('--pretrained_model', default=None, help='path to pretrained model')
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--mini_batch_size', default=32, type=int, help='mini batch size')
parser.add_argument('--mini_batch_chunk_size', default=32, type=int, help='mini batch size chunk')
parser.add_argument('--max_epochs', default=300, type=int, help='max epochs')
parser.add_argument('--patience', default=5, type=int, help='patience')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
parser.add_argument('--rnn', default=1, type=int, help='number of RNN layers')
parser.add_argument('--embeddings_storage_mode', default='gpu', choices=['none', 'cpu', 'gpu'],
                    help='embeddings storage mode')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--embeddings', nargs='+', help='list of embeddings, e.g. flair-pl-forward', required=True)
parser.add_argument('--monitor_train', action='store_true', help='evaluate train data after every epoch')
parser.add_argument('--maca_tags', action='store_true', help='add maca tags as OneHot embeddings')
parser.add_argument('--maca_tags2', action='store_true', help='add maca tags as MoreHot embeddings')
parser.add_argument('--space', action='store_true', help='add space as embeddings')
parser.add_argument('--train_initial_hidden_state', action='store_true', help='train_initial_hidden_state')

args = parser.parse_args()

log = logging.getLogger("args")
log.setLevel('INFO')
base_path = args.output_folder
if type(base_path) is str:
    base_path = Path(base_path)
log_handler = add_file_handler(log, base_path / "args.log")
log.addHandler(logging.StreamHandler())

log.info(str(args))

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


columns = {0: 'text', 1: 'space_before', 2: 'lemma', 3:'pos', 4:'tags'}
#sample 10% of train as dev
corpus: Corpus = TSVCorpus('datasets/pos/', columns,
                                    train_file='nkjp.tsv',
                                    test_file='poleval-test-analyzed-merged.spickle.tsv',
                                    dev_file=None)
tag_type = 'pos'

corpus = corpus.downsample(args.downsample)
log.info(str(corpus))

# 3. make the tag dictionary from the corpus
def count_parameters(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)

if args.pretrained_model is None:
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    log.info(str(tag_dictionary.idx2item))




    embedding_types: List[TokenEmbeddings] = [get_embeddings(name) for name in args.embeddings]
    if args.maca_tags:
        embedding_types.append(OneHotEmbeddings(corpus=corpus, field='tags', embedding_length=20))
    if args.space:
        embedding_types.append(OneHotEmbeddings(corpus=corpus, field='space_before', embedding_length=2))
    if args.maca_tags2:
        embedding_types.append(MoreHotEmbeddings(corpus=corpus, field='tags', embedding_length=20))

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    log.info(f'Embeddings size: {embeddings.embedding_length}')
    log.info(f'Embeddings size: {embeddings}')





    log.info(f'Trainable parameters of embeddings: {count_parameters(embeddings)}')
    log.info(f'Non-trainable parameters of embeddings: {count_parameters(embeddings, trainable=False)}')

    # 5. initialize sequence tagger


    tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_size,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=False,
                                            rnn_layers=args.rnn,
                                            train_initial_hidden_state=args.train_initial_hidden_state)
else:
    tagger: SequenceTagger = SequenceTagger.load(args.pretrained_model)


log.info(f'Trainable parameters of whole model: {count_parameters(tagger)}')
log.info(f'Non-trainable parameters of whole model: {count_parameters(tagger, trainable=False)}')

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus, use_tensorboard=True)

# 7. start training
trainer.train(
    args.output_folder,
    learning_rate=args.learning_rate,
    mini_batch_size=args.mini_batch_size,
    mini_batch_chunk_size=args.mini_batch_chunk_size,
    max_epochs=args.max_epochs,
    min_learning_rate=1e-6,
    shuffle=True,
    anneal_factor=0.5,
    patience=args.patience,
    num_workers=args.num_workers,
    embeddings_storage_mode=args.embeddings_storage_mode,
    monitor_test=True,
    monitor_train=args.monitor_train,
    save_final_model=False)
