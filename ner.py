import random
from argparse import ArgumentParser

import numpy as np
import torch
from flair.data import Corpus
# from flair.datasets import WNUT_17
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BytePairEmbeddings
from typing import List

# 1. get the corpus
import flair.datasets

parser = ArgumentParser(description='Train')
# parser.add_argument('--train_corpus', default='../inforex-data-conversion/tmp/conll_full/train_files.conll', help='path to train JSONL corpus')
# parser.add_argument('--test_corpus', default='../inforex-data-conversion/tmp/conll_full/test_files.conll', help='path to test JSONL corpus')

# parser.add_argument('--forward_model_path', default='polish-forward', help='Flair forward model')
# parser.add_argument('--backward_model_path', default='polish-backward', help='Flair backward model')

parser.add_argument('task', choices=['wikiner', 'ud_pos', 'ud_upos'], help='task')
parser.add_argument('--hidden_size', default=256, type=int, help='size of embedding projection')
parser.add_argument('--downsample', default=1.0, type=float, help='downsample ratio')
parser.add_argument('--output_folder', default='dot1', help='output folder for log and model')
# parser.add_argument('--dev_ratio', default=0.2, type=float, help='dev data ratio of train data')
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--mini_batch_size', default=32, type=int, help='mini batch size')
parser.add_argument('--mini_batch_chunk_size', default=32, type=int, help='mini batch size chunk')
parser.add_argument('--max_epochs', default=100, type=int, help='max epochs')
parser.add_argument('--patience', default=5, type=int, help='patience')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
parser.add_argument('--embeddings_storage_mode', default='gpu', choices=['none', 'cpu', 'gpu'],
                    help='embeddings storage mode')
parser.add_argument('--seed', default=0, type=int, help='seed')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


if args.task == 'wikiner':
    corpus: Corpus = flair.datasets.WIKINER_POLISH()
    tag_type = 'ner'
elif args.task == 'ud_upos':
    corpus: Corpus = flair.datasets.UD_POLISH()
    tag_type = 'upos'
    # lemma (2), upos(3), pos(4), dependency(7)
    # 4       lewica  lewica  NOUN    subst:sg:nom:f  Case=Nom|Gender=Fem|Number=Sing 3       nsubj   3:nsubj SpaceAfter=No
elif args.task == 'ud_pos':
    corpus: Corpus = flair.datasets.UD_POLISH()
    tag_type = 'pos'

corpus = corpus.downsample(args.downsample)
print(corpus)


# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    BytePairEmbeddings('pl')
    # WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_size,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

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
    monitor_train=True,
    save_final_model=False)

# 8. plot weight traces (optional)
from flair.visual.training_curves import Plotter

plotter = Plotter()
plotter.plot_weights('resources/taggers/example-ner/weights.txt')
