import numpy as np
import torch
from tqdm.auto import trange

# converted from sentence-transformers to embed already tokenized text instead of raw text
def token_embedding(
    model,
    input_ids,
    attention_masks,
    batch_size,
    convert_to_tensor=True,
    convert_to_numpy=False,
    show_progress_bar=True,
    normalize_embeddings=True,
):
    model.eval()
    output_value = "sentence_embedding"

    all_embeddings = []
    for start_index in trange(
        0,
        len(input_ids),
        batch_size,
        desc="Generating sentence embeddings",
        disable=not show_progress_bar,
    ):
        end_index = start_index + batch_size
        features = {
            "input_ids": torch.tensor(input_ids[start_index:end_index], device="cuda"),
            "attention_mask": torch.tensor(
                attention_masks[start_index:end_index], device="cuda"
            ),
        }
        with torch.no_grad():
            out_features = model.forward(features)
            embeddings = out_features[output_value]
            embeddings = embeddings.detach()
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
            if convert_to_numpy:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

    if convert_to_tensor:
        all_embeddings = torch.stack(all_embeddings)
    elif convert_to_numpy:
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

    return all_embeddings
