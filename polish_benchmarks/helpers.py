import math
from functools import lru_cache
from typing import List

import flair
import torch
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings


class ConstEmbeddings(TokenEmbeddings):
    """Constant embedding of length 1 with value 1.0 (for testing)."""

    def __init__(self):
        self.name: str = 'const'
        self.static_embeddings = True

        self.__embedding_length: int = 1
        self.position_embedding = torch.tensor([1], device=flair.device, dtype=torch.float)
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def get_cached_vec(self, token_idx: str) -> torch.Tensor:
        return self.position_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token.set_embedding(self.name, self.position_embedding)

        return sentences

    def __str__(self):
        return self.name


class PositionalEmbeddings(TokenEmbeddings):
    """Positional embeddings (for siamese networks with cosine similarity)."""

    def __init__(self, max_position: int = 10):
        """
        :param max_position: Further positions will be clipped to this value.
        """
        self.name: str = 'positional'
        self.static_embeddings = True

        self.__embedding_length: int = 2
        self.max_position=max_position
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @lru_cache(maxsize=1000, typed=False)
    def get_cached_vec(self, token_idx: str) -> torch.Tensor:
        position_value = min(token_idx, self.max_position) * math.pi / 2 / self.max_position
        position_embedding = torch.tensor([math.cos(position_value), math.sin(position_value)], device=flair.device,
                                dtype=torch.float)
        return position_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                position_embedding = self.get_cached_vec(token_idx)
                token.set_embedding(self.name, position_embedding)

        return sentences

    def __str__(self):
        return self.name
