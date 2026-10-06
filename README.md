# Construction PPE Live Monitor

Live webcam detection using the trained YOLO model and the 11 classes in `construction-ppe/data.yaml`.

The camera window shows YOLO boxes with class labels and confidence scores, plus a generic warning when PPE is missing. Voice warnings are queued in the background and do not block camera processing. Press `Q` in the camera window to exit.

## Run

From the project folder, activate the existing virtual environment and start the monitor:

```powershell
.\venv\Scripts\Activate.ps1
python .\live_camera.py
```

The default weights are `runs/detect/outputs/factory_safety-8/weights/best.pt`. To use another trained checkpoint or camera:

```powershell
python .\live_camera.py --model .\path\to\best.pt --camera 0 --confidence 0.10
```

Install dependencies in a new environment with `pip install -r requirements.txt`. A webcam and Windows text-to-speech voice are needed for the complete live/voice experience. If the voice engine is unavailable, visual warnings still work.