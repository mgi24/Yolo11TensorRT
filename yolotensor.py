#USING MODEL IS THE SAME AS NORMAL YOLO
from ultralytics import YOLO

model = YOLO("RTX2060/yolo11l.engine", task="detect")
result = model.predict("man.jpg", save=True)
