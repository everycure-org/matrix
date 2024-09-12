from locust import HttpUser, task, between
import random

MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
INPUT = "The patient presents with fever and cough"


def get_payload():
    num = random.randrange(100)
    payload = {"model": MODEL, "input": [INPUT for i in range(num)]}
    # sometimes we give just the text itself
    if len(payload.get("input")) == 1:
        payload["input"] = INPUT

    return payload


class EmbeddingsUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task
    def get_embedding(self):
        payload = get_payload()
        self.client.post("/v1/embeddings", json=payload)
