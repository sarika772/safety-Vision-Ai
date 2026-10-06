import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# -----------------------------
# Face recognition models
# -----------------------------
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True
)

resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval()
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# -----------------------------
# Face recognition models
# -----------------------------
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True
)

resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval()

# -----------------------------
# Load registered Navya face
# -----------------------------
known_embedding = torch.load(
    r".\worker_data\Navya_embedding.pt",
    weights_only=True
)

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    pil_image = Image.fromarray(rgb_frame)

    # Detect faces
    faces = mtcnn(pil_image)

    if faces is not None:

        # If only one face, make it a batch
        if len(faces.shape) == 3:
            faces = faces.unsqueeze(0)

        # Generate embeddings
        embeddings = resnet(faces).detach()

        for embedding in embeddings:

            # Compare with registered Navya
            similarity = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0),
                known_embedding
            ).item()

            if similarity > 0.60:
                name = "Navya"
            else:
                name = "Unknown"

            # Display name
            cv2.putText(
                frame,
                f"{name} ({similarity:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    else:

        cv2.putText(
            frame,
            "No face detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show camera
    cv2.imshow("Worker Face Recognition", frame)

    # Q → quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# -----------------------------
# Load registered Navya face
# -----------------------------
known_embedding = torch.load(
    r".\worker_data\Navya_embedding.pt",
    weights_only=True
)

# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    pil_image = Image.fromarray(rgb_frame)

    # Detect faces
    faces = mtcnn(pil_image)

    if faces is not None:

        # If only one face, make it a batch
        if len(faces.shape) == 3:
            faces = faces.unsqueeze(0)

        # Generate embeddings
        embeddings = resnet(faces).detach()

        for embedding in embeddings:

            # Compare with registered Navya
            similarity = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0),
                known_embedding
            ).item()

            if similarity > 0.60:
                name = "Navya"
            else:
                name = "Unknown"

            # Display name
            cv2.putText(
                frame,
                f"{name} ({similarity:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    else:

        cv2.putText(
            frame,
            "No face detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show camera
    cv2.imshow("Worker Face Recognition", frame)

    # Q → quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()