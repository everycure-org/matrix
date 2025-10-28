"""Dynamic settings for run_comparison pipeline."""

RUN_COMPARISON_SETTINGS = {
    "evaluations": [
        {
            "name": "ground_truth_recall_at_n",
            "is_activated": True,
        },
        {
            "name": "negative_recall_at_n",
            "is_activated": True,
        },
        {
            "name": "off_label_recall_at_n",
            "is_activated": True,
        },
        {
            "name": "disease_specific_hit_at_k",
            "is_activated": True,
        },
        {
            "name": "disease_specific_hit_at_k_off_label",
            "is_activated": True,
        },
        {
            "name": "drug_entropy_at_n",
            "is_activated": True,
        },
        {
            "name": "disease_entropy_at_n",
            "is_activated": True,
        },
        {
            "name": "commonality_at_n",
            "is_activated": True,
        },
    ],
}
