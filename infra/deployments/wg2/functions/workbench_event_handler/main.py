# NOTE: This file was partially generated using AI assistance.
import base64
import logging

import functions_framework
from cloudevents.http import CloudEvent


@functions_framework.cloud_event
def pubsub_handler(cloud_event: CloudEvent) -> None:
    """Cloud Function triggered by a Cloud Pub/Sub event.

    Args:
        cloud_event (CloudEvent): The CloudEvent object containing the Pub/Sub message.
            See https://cloud.google.com/functions/docs/writing/write-event-driven-functions

    Returns:
        None
    """
    try:
        # Extract and decode the Pub/Sub message data
        if cloud_event.data and "message" in cloud_event.data:
            message_data = cloud_event.data["message"].get("data", "")
            if message_data:
                decoded_message = base64.b64decode(message_data).decode()
                logging.info(f"Received message: {decoded_message}")
                print(f"Received message: {decoded_message}")
            else:
                logging.warning("Received empty message data")
        else:
            logging.warning("Received event without message data")

    except Exception as e:
        logging.error(f"Error processing Pub/Sub message: {str(e)}")
        raise
