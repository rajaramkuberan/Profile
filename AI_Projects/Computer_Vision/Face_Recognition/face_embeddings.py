import argparse
import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from dotenv import load_dotenv
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


def configure_logging():
    """
    Configure logging settings.

    Creates a directory named 'logs' if it doesn't exist and sets up logging to write to a file with a name
    based on the current date.

    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d") + ".log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def connect_to_milvus_db():
    """
    Connect to Milvus database.

    Uses environment variables for user, password, host, and port to establish a connection to the Milvus database.

    Returns:
    - bool: True if the connection is successful, False otherwise.

    """
    try:
        connections.connect(
            alias="default",
            user=os.getenv("MILVUS_USER"),
            password=os.getenv("MILVUS_PASSWORD"),
            host=os.getenv("MILVUS_HOST"),
            port=str(os.getenv("MILVUS_PORT")),
        )
        logging.info("Connected to Milvus DB successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Milvus DB: {str(e)}")
        return False


def load_resnet_model():
    """
    Load pre-trained ResNet model.

    Returns:
    - nn.Module: The loaded pre-trained ResNet model if successful, None otherwise.

    """
    try:
        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        logging.info("Loaded pre-trained ResNet model successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load pre-trained ResNet model: {str(e)}")
        return None


def create_embedding(model, image_path):
    """
    Create embeddings for the given image using the provided pre-trained model.

    Parameters:
    - model: The pre-trained model used for generating embeddings.
    - image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Embeddings of the image.

    """
    try:
        image_to_process = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        process_image = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        tensor_image = process_image(image_to_process).unsqueeze(0)
        with torch.no_grad():
            embeddings = model(tensor_image)
        return embeddings.squeeze().numpy()
    except Exception as e:
        logging.error(f"Error: {e}")
        return None


def main(args):
    """
    This function creates embeddings from images and inserts them into the Milvus DB for Face Recognition.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments -> Image Folder and Collection Name.
    """
    # Configure logging
    configure_logging()

    load_dotenv()

    if not connect_to_milvus_db():
        return

    model = load_resnet_model()
    if model is None:
        return

    # Collection configuration
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=512),
    ]

    # Create the schema
    face_schema = CollectionSchema(fields, "Face collection to store embeddings")
    face_collection = Collection(args.collection_name, schema=face_schema)

    # Create index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    try:
        face_collection.create_index("embeddings", index_params)
        logging.info(
            f"Created index for collection '{args.collection_name}' successfully."
        )
    except Exception as e:
        logging.error(
            f"Failed to create index for collection '{args.collection_name}': {e}"
        )
        return

    # Insert data in Milvus collection
    data = []
    for idx, image_file in enumerate(args.image_folder, start=1):
        embeddings = create_embedding(model, image_file)
        if embeddings is not None:
            data.append([idx, os.path.basename(image_file), embeddings])

    face_collection.insert(data)
    logging.info(
        f"Inserted data into '{args.collection_name}' collection successfully."
    )
    face_collection.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load face images, create embeddings, create collection, and insert into Milvus index"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        nargs="+",
        required=True,
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "-c",
        "--collection_name",
        type=str,
        required=True,
        help="Name of the collection in Milvus",
    )
    args = parser.parse_args()

    main(args)
