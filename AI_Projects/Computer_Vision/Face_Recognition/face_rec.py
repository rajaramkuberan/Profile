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
from ultralytics import YOLO


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


def load_face_collection(collection_name):
    """
    Load face collection from MilvusDB.

    Parameters:
    - collection_name (str): The name of the collection to load.

    Returns:
    - Collection: The loaded face collection if successful, None otherwise.

    """
    try:
        collection = Collection(collection_name)
        collection.load()
        logging.info("Loaded face collection from MilvusDB into memory successfully.")
        return collection
    except Exception as e:
        logging.error(f"Failed to load face collection from MilvusDB: {str(e)}")
        return None


def load_yolov8_model(model_path):
    """
    Load YOLOv8 model.

    Parameters:
    - model_path (str): Path to the YOLOv8 model.

    Returns:
    - YOLO: The loaded YOLOv8 model if successful, None otherwise.

    """
    try:
        model_backend = YOLO(model_path)
        logging.info("Loaded YOLOv8 Model successfully.")
        return model_backend
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 Model: {str(e)}")
        return None


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


def load_face_classes():
    """
    Load face classes from file.

    Reads face classes from a file named 'face.txt' and logs each class name.

    Returns:
    - list: List of face classes if successful, None otherwise.

    """
    try:
        with open("face.txt", "r") as class_face:
            data = class_face.read()
            class_list = data.split("\n")
            logging.info("Loaded face classes from file successfully.")
            for index, class_name in enumerate(class_list):
                logging.info(f"Index: {index}, Class Name: {class_name}")
            return class_list
    except Exception as e:
        logging.error(f"Failed to load face classes from file: {str(e)}")
        return None


def generate_embedding(model, crop_image):
    """
    Generate embeddings for the given crop_image using the provided pre-trained model.

    Parameters:
    - model: The pre-trained model used for generating embeddings.
    - crop_image: A numpy array representing the cropped image in BGR format.

    Returns:
    - embeddings: A numpy array representing the generated embeddings.

    """
    try:
        image_to_process = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
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


def main(model_path, collection_name):
    """
    This function orchestrates the execution of various tasks such as configuring logging, connecting to the Milvus database,
    loading the face collection, YOLOv8 model, pre-trained ResNet model, and face classes from file and perform Face Recognition.

    """
    # Configure logging
    configure_logging()

    load_dotenv()

    if not connect_to_milvus_db():
        return

    collection = load_face_collection(args.collection)
    if collection is None:
        return

    model_backend = load_yolov8_model(args.model_path)
    if model_backend is None:
        return

    model = load_resnet_model()
    if model is None:
        return

    class_list = load_face_classes()
    if class_list is None:
        return

    # similarity search metric definition
    search_params = {
        "metric_type": "L2",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10},
    }

    # Open the webcam as the video source
    cap = cv2.VideoCapture("rtsp://")

    if cap.isOpened():
        logging.info("Video loaded successfully.")
    else:
        logging.error("Failed to load video.")

    # Get video source properties
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Log video source information
    logging.info("Resolution: {}x{}".format(w, h))
    logging.info("FPS: {}".format(fps))

    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            logging.error("Error reading frame from camera. Waiting and retrying...")
            time.sleep(5)  # wait for 5 seconds
            cap = cv2.VideoCapture("rtsp://")
            continue  # go back to the beginning of the loop

        count += 1
        if count % 3 != 0:
            continue

        # Using Yolov8 model predict function:
        result = model_backend.predict(frame)
        a = result[0].boxes.data
        px = pd.DataFrame(a.cpu().numpy()).astype("float")

        # Bounding Box List:

        face_list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if "face" in c:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                face_list.append([x1, y1, x2, y2])

        for bbox in face_list:
            x3, y3, x4, y4 = bbox

            crop = frame[y3:y4, x3:x4]

            predict_rec = generate_embedding(model, crop)

            search_result = collection.search(
                data=predict_rec,
                anns_field="embeddings",
                param=search_params,
                limit=5,
                expr=None,
                output_fields=["id", "image_name"],
                consistency_level="Strong",
            )

            # distance between embeddings for each faces
            distances = search_result[0]["distances"]
            image_names = search_result[0]["image_name"]

            found = False
            for distance, image_name in zip(distances, image_names):
                # Check if distance is less than 10
                if distance < 10:
                    # Retrieve corresponding image name
                    result = image_name
                    color = (0, 255, 0)  # Green color
                    found = True
                    break  # Exit loop after finding the first image with distance < 10

            # If no image with distance < 10 is found, face is flagged as "UnAuthorized"
            if not found:
                result = "UnAuthorized"
                color = (0, 0, 255)  # Red color

            # Draw the result on the frame
            text = f"{result}"
            cv2.putText(
                frame, text, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Display the frame
        # cv2.imshow('Face_Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture, Face Collection from memory, and the OpenCV window closed
    cap.release()
    collection.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition Project")
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="Path to the YOLOv8 model"
    )
    parser.add_argument(
        "-c",
        "--collection_name",
        type=str,
        required=True,
        help="MilvusDB Collection to load",
    )
    args = parser.parse_args()

    main(args.model_path, args.collection_name)
