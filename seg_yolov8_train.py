from ultralytics import YOLO

# Load a model
model = YOLO('/home/lab6/ZZHSSDisk/zzh_yolov8/ultralytics/cfg/models/v8/yolov8-obb.yaml', task='obb')  # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data='/home/lab6/ZZHSSDisk/zzh_yolov8/ultralytics/cfg/datasets/DOTAv1.yaml', epochs=70, imgsz=1024, batch=2,
#                       hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
#                       flipud=0.0,fliplr=0.0, mosaic=0.0, mixup=0.0)
results = model.train(data='/home/lab6/ZZHSSDisk/zzh_yolov8/ultralytics/cfg/datasets/DOTAv1.yaml', epochs=70, imgsz=1024, batch=4)