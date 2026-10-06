import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


WORKER_DATA_DIR = r".\worker_data"

mtcnn = MTCNN(
    image_size=160,
    margin=20
)

resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval()


def register_worker(worker_name, image_bytes):

    # ---------------------------------------------
    # Clean worker name
    # ---------------------------------------------

    worker_name = worker_name.strip()

    if not worker_name:
        return False, "Please enter worker name."

    # ---------------------------------------------
    # Create worker_data folder if needed
    # ---------------------------------------------

    os.makedirs(
        WORKER_DATA_DIR,
        exist_ok=True
    )

    # ---------------------------------------------
    # Check duplicate worker
    # ---------------------------------------------

    embedding_path = os.path.join(
        WORKER_DATA_DIR,
        f"{worker_name}_embedding.pt"
    )

    image_path = os.path.join(
        WORKER_DATA_DIR,
        f"{worker_name}.jpeg"
    )

    if os.path.exists(embedding_path):

        return (
            False,
            f"Worker '{worker_name}' is already registered."
        )

    # ---------------------------------------------
    # Read uploaded image
    # ---------------------------------------------

    try:

        image = Image.open(
            image_bytes
        ).convert("RGB")

    except Exception:

        return (
            False,
            "Could not read the uploaded image."
        )

    # ---------------------------------------------
    # Detect face
    # ---------------------------------------------

    face = mtcnn(image)

    if face is None:

        return (
            False,
            "No face detected. Please upload a clear face photo."
        )

    # ---------------------------------------------
    # Create face embedding
    # ---------------------------------------------

    with torch.no_grad():

        embedding = resnet(
            face.unsqueeze(0)
        ).detach()

    embedding = embedding.squeeze(0)

    # ---------------------------------------------
    # Save worker photo
    # ---------------------------------------------

    image.save(
        image_path,
        format="JPEG"
    )

    # ---------------------------------------------
    # Save embedding
    # ---------------------------------------------

    torch.save(
        embedding,
        embedding_path
    )

    return (
        True,
        f"Worker '{worker_name}' registered successfully."
    )