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
    ],
}
