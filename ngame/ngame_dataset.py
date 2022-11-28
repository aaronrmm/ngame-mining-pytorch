import math
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Set, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample
from sentence_transformers.models import Transformer, Pooling
from torch import nn
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm, trange

from ngame.embedding import token_embedding
from ngame.faiss_index import get_cosine_distance_matrix, cluster


# from ngame.embedding import sentence_embedding, token_embedding
# from ngame.faiss_index import cluster, get_cosine_distance_matrix


class NgameCluster:
    def __init__(self, item_indexes: Set[int], positive_label_indexes: Set[int]):
        self.item_indexes: Set[int] = item_indexes
        self.positive_labels: Set[int] = positive_label_indexes


def smart_tokenize(texts: List[str], batch_size: int, tokenizer, max_length):
    ids = []
    masks = []
    for start_index in trange(
        0, len(texts), batch_size, desc="tokenizing in batches for fast embedding"
    ):
        tokenized_texts = tokenizer.batch_encode_plus(
            texts[start_index : start_index + batch_size],
            padding=True,
            truncation="longest_first",
            max_length=max_length,
        )
        ids.extend(tokenized_texts["input_ids"])
        masks.extend(tokenized_texts["attention_mask"])
    return {"input_ids": ids, "attention_mask": masks}


class NgameDataset(IterableDataset):
    def __init__(
        self,
        train_df: pd.DataFrame,
        encoder: Any,
        total_iterations: int,
        minibatch_size: int,
        max_sentence_length: int,
        cluster_size: int,
        embedding_batch_size: int,
        max_distance: float = 0.3,
        iterations_before_reclustering=5,
        column_name_for_texts: str = "text",
        column_name_for_labels: str = "target",
        debug_output_dir: Path = None,
    ):
        assert (
            minibatch_size % cluster_size == 0
        ), f"Minibatch size must be an even multiple of cluster size."

        self.debug_output_dir = debug_output_dir
        if self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        self.cross_encoder = encoder
        train_df = train_df.dropna()
        self.texts = train_df[column_name_for_texts].tolist()
        tokenized_texts = smart_tokenize(
            texts=self.texts,
            batch_size=embedding_batch_size,
            max_length=max_sentence_length,
            tokenizer=encoder.tokenizer,
        )
        self.text_token_ids = tokenized_texts["input_ids"]
        self.text_attention_masks = tokenized_texts["attention_mask"]
        self.labels = (
            train_df[column_name_for_labels].drop_duplicates(keep="first").tolist()
        )
        print("Creating matches...")
        start_time = time.time()
        self.matches: List[int] = generate_match_map(train_df, column_name_for_labels)
        end_time = time.time()
        print(
            f"Creating matches took {end_time-start_time} seconds at {(end_time-start_time)/len(self.matches)} seconds per match."
        )

        print("Tokenizing labels...")
        tokenized_labels = smart_tokenize(
            texts=self.labels,
            batch_size=embedding_batch_size,
            max_length=max_sentence_length,
            tokenizer=encoder.tokenizer,
        )
        self.label_token_ids = tokenized_labels["input_ids"]
        self.label_attention_mask = tokenized_labels["attention_mask"]
        self.batches = []
        self.max_distance = max_distance
        self.current_iteration = 0
        self.total_clusters = math.ceil(len(self.text_token_ids) / cluster_size)
        self.clusters_per_batch = minibatch_size // cluster_size
        self.iterations_before_reclustering = iterations_before_reclustering
        self.minibatch_size = minibatch_size
        self.embedding_batch_size = embedding_batch_size
        self.clusters: List[NgameCluster] = []
        self.text_embeddings: torch.Tensor
        self.label_embeddings: torch.Tensor
        self.total_iterations: int = total_iterations

        # region Set up the model for encoding
        transformer_model = Transformer(self.cross_encoder.config.name_or_path)
        # hack: replace the transformer's model with the model we are training
        transformer_model.auto_model = list(self.cross_encoder.model._modules.values())[
            0
        ]
        transformer_model.tokenizer = None  # ensure we never use this
        pooling_model = Pooling(
            transformer_model.get_word_embedding_dimension(), "mean"
        )
        modules = [transformer_model, pooling_model]
        modules = OrderedDict(
            [(str(idx), module) for idx, module in enumerate(modules)]
        )
        self.encoder = nn.Sequential(modules).to(self.cross_encoder._target_device)
        # endregion Set up the model for encoding

    def __len__(self):
        return self.total_iterations * self.minibatch_size

    def __iter__(self) -> List[InputExample]:
        for step in tqdm(
            range(self.total_iterations),
            total=self.total_iterations,
            desc="NGame iterations",
        ):
            if self.current_iteration % self.iterations_before_reclustering == 0:
                self.update_clusters()
            selected_clusters: List[NgameCluster] = random.sample(
                self.clusters, k=min(len(self.clusters), self.clusters_per_batch)
            )
            selected_label_indexes: Set[int] = set()
            selected_text_indexes: List[int] = []
            for cluster in selected_clusters:
                selected_label_indexes = selected_label_indexes.union(
                    cluster.positive_labels
                )
                selected_text_indexes.extend(cluster.item_indexes)
            difficult_pairs: List[List[int]] = self.find_difficult_pairs(
                selected_text_indexes,
                list(selected_label_indexes),
                max_distance=self.max_distance,
                max_pairs=self.minibatch_size,
            )
            difficult_negatives = [
                InputExample(
                    texts=[self.texts[t], self.labels[l]],
                    label=0,
                )
                for (t, l) in difficult_pairs
                if l != self.matches[t]
            ]
            difficult_text_ids = set([pair[0] for pair in difficult_pairs])
            positives = [
                InputExample(
                    texts=[self.texts[t], self.labels[self.matches[t]]],
                    label=1,
                )
                for t in list(difficult_text_ids)
            ]
            if len(positives) > self.minibatch_size // 2:
                positives = random.sample(positives, k=self.minibatch_size // 2)
            if len(difficult_negatives) > self.minibatch_size - len(positives):
                difficult_negatives = random.sample(
                    difficult_negatives, k=self.minibatch_size - len(positives)
                )
            difficult_examples = difficult_negatives + positives
            if self.debug_output_dir is not None:
                difficult_example_outputs = [
                    {
                        "match": de.label,
                        "text": de.texts[0],
                        "label": de.texts[1],
                    }
                    for de in difficult_examples
                ]
                diff_df = pd.DataFrame(difficult_example_outputs)
                diff_df.to_csv(
                    self.debug_output_dir
                    / f"difficult_examples_{self.current_iteration}_{len(diff_df)}.csv",
                    index=False,
                )
            # sort to ensure positives and negatives are in same batches
            difficult_examples.sort(key=lambda x: x.texts[0])
            # TODO random hash the text and sort on that for some randomness instead of sorting alphabetically every time
            for example in difficult_examples:
                yield example
            self.current_iteration += 1

    def embed_tokens(self, tokens, attention_masks):
        embeddings: torch.Tensor = token_embedding(
            input_ids=tokens,
            attention_masks=attention_masks,
            model=self.encoder,
            batch_size=self.embedding_batch_size,
            convert_to_tensor=False,
            convert_to_numpy=True,
        )
        return embeddings

    def update_embeddings(self) -> None:
        """ Updates the stored embeddings of each text and label input according to the current model parameters"""
        self.text_embeddings = self.embed_tokens(
            self.text_token_ids, self.text_attention_masks
        )
        self.label_embeddings = self.embed_tokens(
            self.label_token_ids, self.label_attention_mask
        )

    def update_clusters(self) -> None:
        self.update_embeddings()
        self.clusters = generate_clusters(
            self.text_embeddings, self.total_clusters, label_ids=self.matches
        )

    def find_difficult_pairs(
        self,
        selected_text_indexes: List[int],
        selected_label_indexes: List[int],
        max_distance: float,
        max_pairs: int,
        most_difficult: bool = False,
    ):
        """create a distance matrix. Select only those below max_distance"""
        text_embeddings = np.stack(
            [self.text_embeddings[index] for index in selected_text_indexes]
        )
        label_embeddings = np.stack(
            [self.label_embeddings[index] for index in selected_label_indexes]
        )
        distance_matrix = get_cosine_distance_matrix(
            m=text_embeddings, n=label_embeddings
        )
        print(
            f"Searching for difficult pairs in a range between {np.min(distance_matrix)} and {np.max(distance_matrix)}"
        )
        # TODO consider a min distance to start with to make things easier for stability
        thresholded_indexes = np.asarray(distance_matrix <= max_distance).nonzero()
        difficult_pair_indexes = [
            [x, y] for x, y in zip(thresholded_indexes[0], thresholded_indexes[1])
        ]
        distances = [distance_matrix[x, y] for (x, y) in difficult_pair_indexes]
        if len(distances) == 0:
            return []
        if most_difficult:
            ascending_order = np.argsort(distances)
            assert distances[ascending_order[0]] == min(distances)
            most_difficult_pair_indexes = [
                difficult_pair_indexes[i] for i in ascending_order[:max_pairs]
            ]
            most_difficult_pairs = [
                [selected_text_indexes[x], selected_label_indexes[y]]
                for (x, y) in most_difficult_pair_indexes
            ]
            return most_difficult_pairs
        else:
            random.shuffle(difficult_pair_indexes)
            difficult_pairs = [
                [selected_text_indexes[x], selected_label_indexes[y]]
                for (x, y) in difficult_pair_indexes[:max_pairs]
            ]
            return difficult_pairs


def generate_clusters(
    embeddings, num_clusters, label_ids: List[int]
) -> List[NgameCluster]:
    cluster_assignments: List[int] = cluster(embeddings, num_clusters).tolist()
    indexes: pd.Series = pd.Series(range(len(embeddings)))
    label_ids: pd.Series = pd.Series(label_ids)
    cluster_index_sets: List[Set[int]] = (
        indexes.groupby(by=cluster_assignments).apply(set).tolist()
    )
    cluster_label_sets: List[Set[int]] = (
        label_ids.groupby(by=cluster_assignments).apply(set).tolist()
    )
    clusters: List[NgameCluster] = [
        NgameCluster(item_indexes=item_list, positive_label_indexes=label_set)
        for item_list, label_set in zip(cluster_index_sets, cluster_label_sets)
    ]
    return clusters


def generate_match_map(df: pd.DataFrame, column_name_for_labels: str) -> List[int]:
    label_ids: List[str] = list(df[column_name_for_labels].unique())
    label_id_map: Dict[str, int] = {
        label: index for index, label in tqdm(enumerate(label_ids))
    }
    match_map: List[int] = [
        label_id_map[text] for text in tqdm(df[column_name_for_labels])
    ]
    return match_map
