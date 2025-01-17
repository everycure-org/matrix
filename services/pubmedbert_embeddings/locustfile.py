import random

from faker import Faker
from locust import HttpUser, between, task
from models import BASELINE_MODEL, CHALLENGER_MODEL

fake = Faker()

models = [BASELINE_MODEL, CHALLENGER_MODEL]


def get_payload():
    inputs = [fake.sentence() for i in range(random.randrange(250))]
    # sometimes we send just a string
    if len(inputs) == 1:
        inputs = inputs[0]

    return {
        "model": random.choice(models),
        "input": inputs,
    }


class EmbeddingsUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task
    def get_embedding(self):
        payload = get_payload()
        self.client.post("/v1/embeddings", json=payload)
