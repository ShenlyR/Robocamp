from ultralytics import YOLO
model =  YOLO('model_asv_2024/bola.pt')
print(model.names)