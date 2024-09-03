from locust import HttpUser, task, between


class EmbeddingsUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task
    def get_embedding(self):
        payload = {
            "input": "The patient presents with fever and cough",
            "model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        }
        self.client.post("/v1/embeddings", json=payload)
