import sys

sys.path.append('E:/zhangzehao/ultralytics_yolov8/ultralytics')

from ultralytics.data.converter import convert_dota_to_yolo_obb

convert_dota_to_yolo_obb('E:/zhangzehao/ultralytics_yolov8/DOTA1_0_splitdata')
