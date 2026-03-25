# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional, Dict, Tuple

import torch

class SimilarityModule(torch.nn.Module):
    """
    Interface enabling interfacing with various similarity functions.

    While the discussions in our initial ICML'24 paper are based on inner products
    for simplicity, we provide this interface (SimilarityModule) to support various
    learned similarities at the retrieval stage, such as MLPs, Factorization Machines
    (FMs), and Mixture-of-Logits (MoL), which we discussed in
    - Revisiting Neural Retrieval on Accelerators (KDD'23), and
    - Retrieval with Learned Similarities (https://arxiv.org/abs/2407.15462).
    """

    @abc.abstractmethod
    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            query_embeddings: (B, input_embedding_dim) x float.
            item_embeddings: (1/B, X, item_embedding_dim) x float.
            **kwargs: Implementation-specific keys/values (e.g.,
                item ids / sideinfo, etc.)

        Returns:
            A tuple of (
                (B, X,) similarity values,
                keyed outputs representing auxiliary losses at training time.
            ).
        """
        pass


class SequentialEncoderWithLearnedSimilarityModule(torch.nn.Module):
    """
    Interface enabling using various similarity functions (besides inner products)
    as part of a sequential encoder/decoder.

    See rails/ for more details.
    """

    def __init__(
        self,
        ndp_module: SimilarityModule,
    ) -> None:
        super().__init__()

        self._ndp_module: SimilarityModule = ndp_module

    @abc.abstractmethod
    def debug_str(
        self,
    ) -> str:
        pass

    def similarity_fn(
        self,
        query_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        item_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        torch._assert(
            len(query_embeddings.size()) == 2, "len(query_embeddings.size()) must be 2"
        )
        torch._assert(len(item_ids.size()) == 2, "len(item_ids.size()) must be 2")
        if item_embeddings is None:
            item_embeddings = self.get_item_embeddings(item_ids)  # pyre-ignore [29]
        torch._assert(
            len(item_embeddings.size()) == 3, "len(item_embeddings.size()) must be 3"
        )

        return self._ndp_module(
            query_embeddings=query_embeddings,  # (B, query_embedding_dim)
            item_embeddings=item_embeddings,  # (1/B, X, item_embedding_dim)
            item_ids=item_ids,
            **kwargs,
        )
