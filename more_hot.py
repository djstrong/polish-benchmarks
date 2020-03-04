from collections import Counter
from typing import List

import flair
import torch
from flair.data import Corpus, Dictionary, Sentence
from flair.embeddings import TokenEmbeddings


def flatten(l):
    return [item for sublist in l for item in sublist]


class MoreHotEmbeddings(TokenEmbeddings):
    """Average of one-hot encoded embeddings."""

    def __init__(
            self,
            corpus: Corpus,
            field: str = "text",
            embedding_length: int = 300,
            min_freq: int = 3,
            separator: str = '_'
    ):

        super().__init__()
        self.name = "more-hot"
        self.static_embeddings = False
        self.min_freq = min_freq
        self.field = field
        self.separator = separator

        tokens = list(map((lambda s: s.tokens), corpus.train))
        tokens = flatten(tokens)

        if field == "text":
            values = list(map((lambda t: t.text.split(separator)), tokens))
        else:
            values = list(map((lambda t: t.get_tag(field).value.split(separator)), tokens))
        values = flatten(values)
        most_common = Counter(values).most_common()

        tokens = []
        for token, freq in most_common:
            if freq < min_freq:
                break
            tokens.append(token)

        self.vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            self.vocab_dictionary.add_item(token)

        # max_tokens = 500
        self.__embedding_length = embedding_length

        print(self.vocab_dictionary.idx2item)
        print(f"vocabulary size of {len(self.vocab_dictionary)}")

        # model architecture
        self.embedding_layer = torch.nn.Embedding(
            len(self.vocab_dictionary), self.__embedding_length
        )
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        one_hot_sentences = []
        values_counts = []
        for i, sentence in enumerate(sentences):

            if self.field == "text":
                context_idxs = [
                    self.vocab_dictionary.get_idx_for_items(t.text.split(self.separator))
                    for t in sentence.tokens
                ]
            else:
                context_idxs = [
                    self.vocab_dictionary.get_idx_for_items(t.get_tag(self.field).value.split(self.separator))
                    for t in sentence.tokens
                ]
            # print(context_idxs)
            # print([len(x) for x in context_idxs])
            # print(flatten(context_idxs))
            values_counts.extend([len(x) for x in context_idxs])
            one_hot_sentences.extend(flatten(context_idxs))

        one_hot_sentences = torch.tensor(one_hot_sentences, dtype=torch.long).to(
            flair.device
        )

        embedded = self.embedding_layer.forward(one_hot_sentences)

        token_index = 0
        embedding_index = 0
        for sentence in sentences:
            for token in sentence:
                values_count = values_counts[token_index]
                # print(values_count, embedded[embedding_index:embedding_index + values_count])
                embedding = torch.mean(embedded[embedding_index:embedding_index + values_count], dim=0)
                # print(embedding)
                # print()
                token.set_embedding(self.name, embedding)
                token_index += 1
                embedding_index += values_count

        return sentences
