from ultralytics import YOLO

# Загружаем предобученную модель сегментации
model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100, #Количество эпох
    imgsz=640, #Размер входных изображений
    batch=8,  #Количество одновременно обрабатываемых изображений за раз
    project="runs",
    name="balloon_train"
)