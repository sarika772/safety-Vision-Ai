from ultralytics import YOLO

model = YOLO(
    r"C:\Users\sarik\OneDrive\Desktop\IFWSP\runs\detect\outputs\factory_safety-8\weights\last.pt"
)

model.train(resume=True)