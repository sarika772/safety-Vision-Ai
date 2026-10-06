import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# =========================================================
# FACE RECOGNITION MODELS
# =========================================================

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True
)

resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval()


# =========================================================
# LOAD REGISTERED WORKERS
# =========================================================

WORKER_DATA_DIR = r".\worker_data"

known_workers = {}

for filename in os.listdir(WORKER_DATA_DIR):

    if filename.endswith("_embedding.pt"):

        worker_name = filename.replace(
            "_embedding.pt",
            ""
        )

        embedding_path = os.path.join(
            WORKER_DATA_DIR,
            filename
        )

        embedding = torch.load(
            embedding_path,
            weights_only=True
        )

        known_workers[worker_name] = embedding


print(
    "Registered workers:",
    list(known_workers.keys())
)


# =========================================================
# RECOGNIZE FACES
# =========================================================

def recognize_faces(image):

    """
    Input:
        OpenCV BGR image

    Output:
        List of recognized faces

        Example:
        [
            {
                "name": "Navya",
                "confidence": 0.82,
                "box": (x1, y1, x2, y2)
            }
        ]
    """

    # BGR → RGB
    rgb_image = image[:, :, ::-1]

    pil_image = Image.fromarray(
        rgb_image
    )

    # Detect faces
    boxes, probabilities = mtcnn.detect(
        pil_image
    )

    if boxes is None:
        return []

    recognized_faces = []

    for box, probability in zip(
        boxes,
        probabilities
    ):

        if probability is None:
            continue

        if probability < 0.90:
            continue

        # Get face crop
        x1, y1, x2, y2 = [
            int(value)
            for value in box
        ]

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        face_crop = pil_image.crop(
            (x1, y1, x2, y2)
        )

        # Resize face
        face_crop = face_crop.resize(
            (160, 160)
        )

        # Convert to tensor
        face_tensor = torch.tensor(
            __import__("numpy").array(
                face_crop
            )
        ).permute(
            2, 0, 1
        ).float() / 255.0

        # Normalize approximately for FaceNet
        face_tensor = (
            face_tensor - 0.5
        ) / 0.5

        # Generate embedding
        with torch.no_grad():

            embedding = resnet(
                face_tensor.unsqueeze(0)
            )

        # Find best matching worker
        best_name = "Unknown"
        best_similarity = 0.0

        for worker_name, known_embedding in known_workers.items():

            similarity = (
                torch.nn.functional.cosine_similarity(
                    embedding,
                    known_embedding
                ).item()
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_name = worker_name

        # Recognition threshold
        if best_similarity < 0.60:

            best_name = "Unknown"

        recognized_faces.append(
            {
                "name": best_name,
                "confidence": best_similarity,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }
        )

    return recognized_faces