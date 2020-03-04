from argparse import ArgumentParser
from pathlib import Path

from flair.datasets import DataLoader

from flair.datasets import ColumnDataset

from tsv import TSVDataset

parser = ArgumentParser(description='Train')

parser.add_argument('model', help='path to model')
parser.add_argument('data', help='path to data')
parser.add_argument('output', help='path to output XCES')

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

# print(f'Trainable parameters of whole model: {count_parameters(tagger)}')
# print(f'Non-trainable parameters of whole model: {count_parameters(tagger, trainable=False)}')

results = tagger.predict(data)


def escape_xml(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace('\'',
                                                                                                            '&apos;')
omg=['spraw','tym','innymi', 'przykład','zwany','między','tytułem','tak','cetera','scriptum','znaczy','rok']

def results_to_xces_str(sentences):
    result_str = []
    result_str += ('<?xml version="1.0" encoding="UTF-8"?>',
                   '<!DOCTYPE cesAna SYSTEM "xcesAnaIPI.dtd">',
                   '<cesAna xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0" type="lex disamb">',
                   '<chunkList>')
    # for sentence in :
    result_str += (' <chunk type="p">', )
    for sentence in sentences:
        result_str += ('  <chunk type="s">',)
        for token in sentence:
            form = token.text
            space_before = token.get_tag('space_before').value
            tag = token.get_tag('pos').value
            # print(tag)
            if tag in omg:
                tag='interp'

            if space_before == '0':
                result_str += ('   <ns/>',)
            result_str += ('   <tok>',)
            result_str += ('    <orth>%s</orth>' % escape_xml(form),)
            # for lemma in token['lemmas']:
            lemma=form
            result_str += ('    <lex disamb="1"><base>%s</base><ctag>%s</ctag></lex>' % (escape_xml(lemma),
                                                                                             tag),)
            result_str += ('   </tok>',)
        result_str += ('  </chunk>',)
    result_str += (' </chunk>',)

    result_str += ('</chunkList>',
                   '</cesAna>')
    return '\n'.join(result_str)


xces = results_to_xces_str(results)
# print(xces)

f=open(args.output, 'w')
f.write(xces)
f.close()

# for sentence in results:
#     for token in sentence:
#         form = token.text
#         space_before=token.get_tag('space_before').value
#         tag=token.get_tag('pos').value
#         print(form, space_before, tag)

# eval_result, loss = tagger.evaluate(
#     DataLoader(
#         data,
#         batch_size=32,
#         num_workers=1,
#     )
# )
#
# print(eval_result.detailed_results)
# print(eval_result.main_score)
