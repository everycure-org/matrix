"""Calculate embeddings."""
import spacy
import time
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import joblib
from openai import OpenAI
from tqdm import tqdm

PATH = "feature_list.pkl"
BATCH_SIZE = 512


def infer_pubmed(feat_list: list):
    """Calculate embeddings with pubmed."""
    feat_list = feat_list
    t1 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    ).to("mps")
    all_embeddings = []
    for start_id in tqdm(range(0, len(feat_list), BATCH_SIZE)):
        end_id = start_id + BATCH_SIZE
        inputs = tokenizer(
            feat_list[start_id:end_id],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = inputs.to("mps")
        with torch.no_grad():
            embeddings = model(
                **inputs, output_hidden_states=True, return_dict=True
            ).pooler_output
        all_embeddings.append(embeddings.cpu())
    print(len(all_embeddings))
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    print(len(all_embeddings))
    joblib.dump(all_embeddings, "scratch/pubmedbert_sample_embed.joblib")
    t2 = time.time()
    print("done and saved in ", str(t2 - t1))


def infer_openai(feat_list: list):
    """Calculate embeddings with openai api."""
    t1 = time.time()
    all_embeddings = []
    for start_id in tqdm(range(0, len(feat_list), BATCH_SIZE)):
        end_id = start_id + BATCH_SIZE
        subfeature = feat_list[start_id:end_id]
        embeddings = client.embeddings.create(
            input=np.array(subfeature), model="text-embedding-3-small", dimensions=768
        )
        all_embeddings.append(embeddings)
    final_list = []
    for batch in all_embeddings:
        for embedding in batch.data:
            final_list.append(embedding.embedding)
    joblib.dump(np.array(emb_list), "scratch/openai_sample_embed.joblib")
    t2 = time.time()
    print("done and saved in ", str(t2 - t1))


def infer_spacy(feat_list: list, model: str):
    """Calculate embeddings with spacy."""
    t1 = time.time()
    nlp = spacy.load(model, disable=["NER"])
    all_embeddings = []
    for feature in tqdm(feat_list):
        doc = nlp(feature)
        all_embeddings.append(doc.vector())
    joblib.dump(np.array(emb_list), f"scratch/spacy_{model}_sample_embed.joblib")
    t2 = time.time()
    print("done and saved in ", str(t2 - t1))


if __name__ == "__main__":
    features = joblib.load("sample_features.joblib")
    print("pubmed")
    infer_pubmed(features)
    print("openai")
    infer_openai(features)
    print("spacy")
    infer_spacy(features, "en_core_web_md")
    print("scispacy")
    infer_spacy(features, "en_core_sci_md")
