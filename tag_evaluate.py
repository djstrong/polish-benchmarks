import sys
from argparse import ArgumentParser
from pathlib import Path

from flair.datasets import DataLoader

from flair.datasets import ColumnDataset

from tsv import TSVDataset

parser = ArgumentParser(description='Train')

parser.add_argument('model', help='path to model')
parser.add_argument('data', help='path to data')

args = parser.parse_args()

columns = {0: 'text', 1: 'space_before', 2: 'lemma', 3: 'pos', 4: 'tags'}

data = TSVDataset(Path(args.data), columns)


def count_parameters(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)


from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger.load(args.model)

print(f'Trainable parameters of whole model: {count_parameters(tagger)}')
print(f'Non-trainable parameters of whole model: {count_parameters(tagger, trainable=False)}')

eval_result, loss = tagger.evaluate(
    DataLoader(
        data,
        batch_size=32,
        num_workers=1,
    )
)

print(eval_result.detailed_results)
print(eval_result.main_score)
