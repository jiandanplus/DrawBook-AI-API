import argparse
import cv2
import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import sys
import json

# 获取项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录到sys.path，确保可以导入library模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 可选导入library.utils，如果不存在则使用备用方案
try:
    from library.utils import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    # 如果无法导入library.utils，则使用基本的日志配置
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("无法导入library.utils，使用基本日志配置")

# 可选导入MTCNN，如果不存在则使用备用方案
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning("无法导入MTCNN库，将使用OpenCV检测器")

# 可选导入YOLOv8-face
try:
    import torch
    from ultralytics import YOLO
    # 为PyTorch 2.6+添加安全全局变量支持
    try:
        from ultralytics.nn.tasks import DetectionModel
        from torch.nn.modules.container import Sequential
        torch.serialization.add_safe_globals([DetectionModel, Sequential])
    except Exception:
        pass  # 如果无法导入或添加安全全局变量，则忽略
    
    # 保存原始的torch.load函数
    original_torch_load = torch.load
    
    # 创建一个新的torch.load函数，将weights_only默认设置为False
    def patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=True, **pickle_load_args):
        return original_torch_load(f, map_location, pickle_module, weights_only=False, **pickle_load_args)
    
    # 替换torch.load函数
    torch.load = patched_torch_load
    
    YOLOV8_AVAILABLE = True
except ImportError:
    YOLOV8_AVAILABLE = False
    logger.warning("无法导入YOLOv8库，将使用其他检测器")

# 可选导入YOLOv5-face
try:
    import torch
    from ultralytics import YOLO
    YOLOV5_AVAILABLE = True
except ImportError:
    YOLOV5_AVAILABLE = False
    logger.warning("无法导入YOLOv5库，将使用其他检测器")

# 可选导入BlazeFace
try:
    import cv2.dnn
    BLAZEFACE_AVAILABLE = True
except ImportError:
    BLAZEFACE_AVAILABLE = False
    logger.warning("无法导入BlazeFace相关库，将使用其他检测器")

# 可选导入Ultra-Light-Fast-Generic-Face-Detector
try:
    import onnxruntime
    ULTRA_FACE_AVAILABLE = True
except ImportError:
    ULTRA_FACE_AVAILABLE = False
    logger.warning("无法导入Ultra-Light-Fast-Generic-Face-Detector相关库，将使用其他检测器")


class FaceCropper:
    def __init__(self, min_neighbors=5, scale_factor=1.1, min_size=(30, 30), confidence_threshold=0.5, 
                 use_mtcnn=False, use_yolov8=False, use_blazeface=False, use_ultraface=False,
                 strict_face_filter=True, face_detector=None):
        """
        初始化人脸裁剪器
        
        Args:
            min_neighbors (int): 人脸检测参数，值越大检测越严格
            scale_factor (float): 人脸检测参数，检测窗口缩放因子
            min_size (tuple): 人脸检测参数，最小人脸尺寸
            confidence_threshold (float): DNN检测器置信度阈值
            use_mtcnn (bool): 是否使用MTCNN检测器（如果可用）
            use_yolov8 (bool): 是否使用YOLOv8-face检测器（如果可用）
            use_blazeface (bool): 是否使用BlazeFace检测器（如果可用）
            use_ultraface (bool): 是否使用Ultra-Light-Fast-Generic-Face-Detector（如果可用）
            strict_face_filter (bool): 是否使用严格的人脸面积过滤
            face_detector (str): 人脸检测算法名称，可选值: "yolov8", "mtcnn", "blazeface", "ultraface", "opencv_dnn", "haar_cascade"
        """
        # 加载OpenCV的人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.confidence_threshold = confidence_threshold
        self.use_mtcnn = use_mtcnn and MTCNN_AVAILABLE
        self.use_yolov8 = use_yolov8 and YOLOV8_AVAILABLE
        self.use_blazeface = use_blazeface and BLAZEFACE_AVAILABLE
        self.use_ultraface = use_ultraface and ULTRA_FACE_AVAILABLE
        self.strict_face_filter = strict_face_filter
        self.face_detector = face_detector
        
        # 如果启用了特定的检测器，确保它被正确设置
        if face_detector:
            self.use_yolov8 = (face_detector == "yolov8") and YOLOV8_AVAILABLE
            self.use_mtcnn = (face_detector == "mtcnn") and MTCNN_AVAILABLE
            self.use_blazeface = (face_detector == "blazeface") and BLAZEFACE_AVAILABLE
            self.use_ultraface = (face_detector == "ultraface") and ULTRA_FACE_AVAILABLE
        
        # 如果启用且可用，初始化MTCNN检测器
        if self.use_mtcnn and MTCNN_AVAILABLE:
            self.mtcnn_detector = MTCNN()
            
        # 如果启用且可用，初始化YOLOv8-face检测器
        if self.use_yolov8 and YOLOV8_AVAILABLE:
            try:
                # 只从models文件夹加载face_yolov8m.pt模型文件
                model_path = os.path.join(PROJECT_ROOT, 'models', 'face_yolov8m.pt')
                # 检查模型文件是否存在
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")
                self.yolov8_model = YOLO(model_path)
                logger.info(f"成功加载人脸检测模型: {model_path}")
                self.yolov8_model.conf = self.confidence_threshold
            except Exception as e:
                logger.warning(f"无法加载YOLOv8-face模型: {e}")
                self.use_yolov8 = False
                
        # 如果启用且可用，初始化BlazeFace检测器
        if self.use_blazeface and BLAZEFACE_AVAILABLE:
            try:
                # 尝试加载BlazeFace模型
                self.blazeface_model = cv2.dnn.readNetFromTensorflow('blazeface_back_model.pb', 'blazeface_back_model.pbtxt')
            except Exception as e:
                logger.warning(f"无法加载BlazeFace模型: {e}")
                self.use_blazeface = False
                
        # 如果启用且可用，初始化Ultra-Light-Fast-Generic-Face-Detector
        if self.use_ultraface and ULTRA_FACE_AVAILABLE:
            try:
                # 尝试加载UltraFace模型
                self.ultraface_model = onnxruntime.InferenceSession('version-RFB-320.onnx')
            except Exception as e:
                logger.warning(f"无法加载UltraFace模型: {e}")
                self.use_ultraface = False
    
    def detect_faces_mtcnn(self, image):
        """
        使用MTCNN检测人脸
        
        Args:
            image (numpy.ndarray): 图像数据
            
        Returns:
            list: 包含所有人脸位置信息的列表 [(x, y, w, h, confidence), ...]
        """
        if not MTCNN_AVAILABLE:
            return []
        
        try:
            # MTCNN检测人脸
            results = self.mtcnn_detector.detect_faces(image)
            
            faces = []
            for result in results:
                confidence = result['confidence']
                if confidence > self.confidence_threshold:
                    x, y, w, h = result['box']
                    faces.append((x, y, w, h, confidence))
            
            # 按置信度排序
            faces.sort(key=lambda x: x[4], reverse=True)
            return faces
        except Exception as e:
            logger.warning(f"MTCNN人脸检测出错: {e}")
            return []
    
    def detect_faces_yolov8(self, image):
        """
        使用YOLOv8-face检测人脸
        
        Args:
            image (numpy.ndarray): 图像数据
            
        Returns:
            list: 包含所有人脸位置信息的列表 [(x, y, w, h, confidence), ...]
        """
        if not YOLOV8_AVAILABLE or not self.use_yolov8:
            return []
        
        try:
            # YOLOv8-face检测人脸
            results = self.yolov8_model(image)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = box.conf.item()
                        if conf > self.confidence_threshold:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            w = x2 - x1
                            h = y2 - y1
                            faces.append((int(x1), int(y1), int(w), int(h), conf))
            
            # 按置信度排序
            faces.sort(key=lambda x: x[4], reverse=True)
            return faces
        except Exception as e:
            logger.warning(f"YOLOv8-face人脸检测出错: {e}")
            return []
    
    def detect_faces_blazeface(self, image):
        """
        使用BlazeFace检测人脸
        
        Args:
            image (numpy.ndarray): 图像数据
            
        Returns:
            list: 包含所有人脸位置信息的列表 [(x, y, w, h, confidence), ...]
        """
        if not BLAZEFACE_AVAILABLE or not self.use_blazeface:
            return []
        
        try:
            # BlazeFace检测人脸
            blob = cv2.dnn.blobFromImage(image, 1.0, (128, 128), (104, 117, 123))
            self.blazeface_model.setInput(blob)
            detections = self.blazeface_model.forward()
            
            faces = []
            (h, w) = image.shape[:2]
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # 过滤低置信度的检测结果
                if confidence > self.confidence_threshold:
                    # 计算边界框坐标
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # 确保边界框在图像范围内
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # 转换为 (x, y, w, h) 格式
                    width = endX - startX
                    height = endY - startY
                    
                    faces.append((startX, startY, width, height, confidence))
            
            # 按置信度排序
            faces.sort(key=lambda x: x[4], reverse=True)
            return faces
        except Exception as e:
            logger.warning(f"BlazeFace人脸检测出错: {e}")
            return []
    
    def detect_faces_ultraface(self, image):
        """
        使用Ultra-Light-Fast-Generic-Face-Detector检测人脸
        
        Args:
            image (numpy.ndarray): 图像数据
            
        Returns:
            list: 包含所有人脸位置信息的列表 [(x, y, w, h, confidence), ...]
        """
        if not ULTRA_FACE_AVAILABLE or not self.use_ultraface:
            return []
        
        try:
            # UltraFace检测人脸
            # 预处理图像
            image_height, image_width = image.shape[:2]
            image_resized = cv2.resize(image, (320, 240))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_transposed = np.transpose(image_rgb, [2, 0, 1])
            image_processed = np.expand_dims(image_transposed, axis=0).astype(np.float32)
            
            # 运行推理
            input_name = self.ultraface_model.get_inputs()[0].name
            results = self.ultraface_model.run(None, {input_name: image_processed})
            
            faces = []
            boxes = results[0][0]
            for box in boxes:
                confidence = box[2]
                if confidence > self.confidence_threshold:
                    x1 = int(box[0] * image_width)
                    y1 = int(box[1] * image_height)
                    x2 = int(box[2] * image_width)
                    y2 = int(box[3] * image_height)
                    w = x2 - x1
                    h = y2 - y1
                    faces.append((x1, y1, w, h, confidence))
            
            # 按置信度排序
            faces.sort(key=lambda x: x[4], reverse=True)
            return faces
        except Exception as e:
            logger.warning(f"UltraFace人脸检测出错: {e}")
            return []
    
    def detect_faces_cnn(self, image):
        """
        使用OpenCV的DNN模块检测人脸 (基于Caffe模型)
        
        Args:
            image (numpy.ndarray): 图像数据
            
        Returns:
            list: 包含所有人脸位置信息的列表 [(x, y, w, h, confidence), ...]
        """
        # 检查模型文件是否存在
        prototxt_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
            logger.debug("DNN模型文件不存在，使用Haar级联检测器")
            return []
        
        try:
            # 加载模型
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            
            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # 过滤低置信度的检测结果
                if confidence > self.confidence_threshold:
                    # 计算边界框坐标
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # 确保边界框在图像范围内
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # 转换为 (x, y, w, h) 格式
                    width = endX - startX
                    height = endY - startY
                    
                    faces.append((startX, startY, width, height, confidence))
            
            # 按置信度排序
            faces.sort(key=lambda x: x[4], reverse=True)
            return faces
        except Exception as e:
            logger.warning(f"DNN人脸检测出错: {e}")
            return []
    
    def detect_faces(self, image_path):
        """
        检测图片中的人脸位置
        
        Args:
            image_path (str): 图片路径
            
        Returns:
            list: 包含所有人脸位置信息的列表 [(x, y, w, h), ...]
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"无法读取图片: {image_path}")
            return []
        
        # 获取图片尺寸
        img_height, img_width = img.shape[:2]
        img_area = img_width * img_height
        
        # 根据strict_face_filter参数设置过滤条件
        if self.strict_face_filter:
            min_ratio, max_ratio = 0.005, 0.5  # 0.5%到50%
        else:
            min_ratio, max_ratio = 0.0005, 0.8  # 0.05%到80%，更宽松的条件
        
        # 如果指定了特定的检测器，则只使用该检测器
        if self.face_detector:
            if self.face_detector == "yolov8" and self.use_yolov8:
                yolov8_faces = self.detect_faces_yolov8(img)
                return self._filter_faces(yolov8_faces, img_area, min_ratio, max_ratio)
            
            elif self.face_detector == "mtcnn" and self.use_mtcnn:
                mtcnn_faces = self.detect_faces_mtcnn(img)
                return self._filter_faces(mtcnn_faces, img_area, min_ratio, max_ratio)
                
            elif self.face_detector == "blazeface" and self.use_blazeface:
                blazeface_faces = self.detect_faces_blazeface(img)
                return self._filter_faces(blazeface_faces, img_area, min_ratio, max_ratio)
                
            elif self.face_detector == "ultraface" and self.use_ultraface:
                ultraface_faces = self.detect_faces_ultraface(img)
                return self._filter_faces(ultraface_faces, img_area, min_ratio, max_ratio)
                
            elif self.face_detector == "opencv_dnn":
                # 使用CNN检测器
                cnn_faces = self.detect_faces_cnn(img)
                return self._filter_faces(cnn_faces, img_area, min_ratio, max_ratio)
                
            elif self.face_detector == "haar_cascade":
                # 使用Haar级联检测器
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size
                )
                # 转换为列表格式
                if isinstance(faces, np.ndarray):
                    faces_list = [(x, y, w, h) for (x, y, w, h) in faces.tolist()]
                else:
                    faces_list = []
                # 过滤掉面积过小或过大的人脸（可能是误检）
                return self._filter_faces_by_area(faces_list, img_area, min_ratio, max_ratio)
        
        # 向后兼容：如果没有指定特定检测器，则使用原来的优先级逻辑
        # 优先使用YOLOv8-face检测器（如果启用且可用）
        if self.use_yolov8:
            yolov8_faces = self.detect_faces_yolov8(img)
            if yolov8_faces:
                faces = [(x, y, w, h) for (x, y, w, h, _) in yolov8_faces]
                return self._filter_faces_by_area(faces, img_area, min_ratio, max_ratio)
        
        # 使用BlazeFace检测器（如果启用且可用）
        if self.use_blazeface:
            blazeface_faces = self.detect_faces_blazeface(img)
            if blazeface_faces:
                faces = [(x, y, w, h) for (x, y, w, h, _) in blazeface_faces]
                return self._filter_faces_by_area(faces, img_area, min_ratio, max_ratio)
        
        # 使用UltraFace检测器（如果启用且可用）
        if self.use_ultraface:
            ultraface_faces = self.detect_faces_ultraface(img)
            if ultraface_faces:
                faces = [(x, y, w, h) for (x, y, w, h, _) in ultraface_faces]
                return self._filter_faces_by_area(faces, img_area, min_ratio, max_ratio)
        
        # 使用MTCNN检测器（如果启用且可用）
        if self.use_mtcnn:
            mtcnn_faces = self.detect_faces_mtcnn(img)
            if mtcnn_faces:
                faces = [(x, y, w, h) for (x, y, w, h, _) in mtcnn_faces]
                return self._filter_faces_by_area(faces, img_area, min_ratio, max_ratio)
        
        # 使用CNN检测器作为备选
        cnn_faces = self.detect_faces_cnn(img)
        if cnn_faces:
            faces = [(x, y, w, h) for (x, y, w, h, _) in cnn_faces]
            return self._filter_faces_by_area(faces, img_area, min_ratio, max_ratio)
        
        # 转换为灰度图以提高检测效果
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用Haar级联检测器作为最后备选
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        # 转换为列表格式
        if isinstance(faces, np.ndarray):
            faces_list = [(x, y, w, h) for (x, y, w, h) in faces.tolist()]
        else:
            faces_list = []
        
        # 过滤掉面积过小或过大的人脸（可能是误检）
        return self._filter_faces_by_area(faces_list, img_area, min_ratio, max_ratio)
    
    def _filter_faces(self, faces_with_conf, img_area, min_ratio, max_ratio):
        """
        过滤带置信度的人脸检测结果
        
        Args:
            faces_with_conf: 带置信度的人脸检测结果 [(x, y, w, h, confidence), ...]
            img_area: 图像面积
            min_ratio: 最小人脸面积比例
            max_ratio: 最大人脸面积比例
            
        Returns:
            list: 过滤后的人脸坐标 [(x, y, w, h), ...]
        """
        if not faces_with_conf:
            return []
            
        # 只返回坐标信息，去掉置信度
        faces = [(x, y, w, h) for (x, y, w, h, _) in faces_with_conf]
        return self._filter_faces_by_area(faces, img_area, min_ratio, max_ratio)
    
    def _filter_faces_by_area(self, faces, img_area, min_ratio, max_ratio):
        """
        根据面积过滤人脸检测结果
        
        Args:
            faces: 人脸坐标列表 [(x, y, w, h), ...]
            img_area: 图像面积
            min_ratio: 最小人脸面积比例
            max_ratio: 最大人脸面积比例
            
        Returns:
            list: 过滤后的人脸坐标 [(x, y, w, h), ...]
        """
        filtered_faces = []
        for (x, y, w, h) in faces:
            face_area = w * h
            # 根据strict_face_filter参数调整过滤条件
            if img_area * min_ratio <= face_area <= img_area * max_ratio:
                filtered_faces.append((x, y, w, h))
        return filtered_faces
    
    def detect_largest_face(self, image_path):
        """
        检测图片中占比最大的人脸
        
        Args:
            image_path (str): 图片路径
            
        Returns:
            tuple: 最大人脸的位置信息 (x, y, w, h)，如果没有检测到人脸则返回 None
        """
        faces = self.detect_faces(image_path)
        
        if not faces:
            return None
        
        # 计算每个人脸的面积，选择面积最大的人脸
        largest_face = None
        largest_area = 0
        
        for (x, y, w, h) in faces:
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, w, h)
        
        return largest_face
    
    def crop_face(self, image, face_coords, crop_size=(512, 512), expand_ratio=1.8):
        """
        根据人脸位置裁切图片
        
        Args:
            image (PIL.Image): 图片对象
            face_coords (tuple): 人脸坐标 (x, y, w, h)
            crop_size (tuple): 裁切后的图片尺寸 (width, height)
            expand_ratio (float): 扩展比例，控制裁切范围，增大到1.8以包含更多头部区域
            
        Returns:
            PIL.Image: 裁切后的人脸图片
        """
        img_width, img_height = image.size
        
        x, y, w, h = face_coords
        
        # 计算人脸中心点
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # 使用人脸最大尺寸作为基准（确保包含完整的头部）
        face_max_size = max(w, h)
        
        # 计算裁切区域的尺寸（基于人脸最大尺寸并应用扩展比例）
        crop_width = int(face_max_size * expand_ratio)
        crop_height = int(face_max_size * expand_ratio)
        
        # 确保裁切尺寸不超过原始图片尺寸
        crop_width = min(crop_width, img_width)
        crop_height = min(crop_height, img_height)
        
        # 计算裁切区域的左上角坐标，确保人脸在中心
        left = max(0, face_center_x - crop_width // 2)
        top = max(0, face_center_y - crop_height // 2)
        right = min(img_width, left + crop_width)
        bottom = min(img_height, top + crop_height)
        
        # 调整坐标以确保裁切区域大小正确，并保持人脸在中心
        if right - left < crop_width:
            left = max(0, right - crop_width)
        if bottom - top < crop_height:
            top = max(0, bottom - crop_height)
            
        # 如果裁切区域超出了图片边界，需要调整使人脸保持在中心
        if left == 0:
            right = min(crop_width, img_width)
        if top == 0:
            bottom = min(crop_height, img_height)
        if right == img_width:
            left = max(0, img_width - crop_width)
        if bottom == img_height:
            top = max(0, img_height - crop_height)
        
        # 裁切图片
        cropped_img = image.crop((left, top, right, bottom))
        
        # 调整到目标尺寸
        cropped_img = cropped_img.resize(crop_size, Image.Resampling.LANCZOS)
        
        return cropped_img
    
    def crop_all_faces_in_image(self, image_path, output_dir=None, crop_size=(512, 512), expand_ratio=1.8, save_flipped=False, filename_prefix="", start_count=0):
        """
        裁切图片中所有检测到的人脸
        
        Args:
            image_path (str): 图片路径
            output_dir (str): 输出目录，如果为None则不保存
            crop_size (tuple): 裁切后的图片尺寸 (width, height)
            expand_ratio (float): 扩展比例，控制裁切范围，增大到1.8以包含更多头部区域
            save_flipped (bool): 是否同时保存水平翻转的图片
            filename_prefix (str): 文件名前缀，用于重命名
            start_count (int): 起始计数，用于全局命名
            
        Returns:
            list: 裁切后的人脸图片列表
        """
        # 检测所有人脸
        faces = self.detect_faces(image_path)
        
        if not faces:
            logger.info(f"未检测到人脸: {image_path}")
            return []
        
        # 读取图片
        img = Image.open(image_path)
        
        cropped_faces = []
        
        # 为每个检测到的人脸创建裁切图片
        for i, face_coords in enumerate(faces):
            cropped_face = self.crop_face(img, face_coords, crop_size, expand_ratio)
            cropped_faces.append(cropped_face)
            
            # 保存图片（如果指定了输出目录）
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # 生成基础文件名
                if filename_prefix:
                    base_name = f"{filename_prefix}_{start_count + i + 1:03d}"
                else:
                    filename = os.path.splitext(os.path.basename(image_path))[0]
                    base_name = f"{filename}_face_{i+1}"
                
                # 保存原始裁切图片
                output_path = os.path.join(output_dir, f"{base_name}.jpg")
                cropped_face.save(output_path)
                logger.info(f"已保存原始裁切图片: {output_path}")
                
                # 如果需要保存水平翻转的图片
                if save_flipped:
                    flipped_face = cropped_face.transpose(Image.FLIP_LEFT_RIGHT)
                    if filename_prefix:
                        flipped_output_path = os.path.join(output_dir, f"{filename_prefix}_{start_count + i + 1 + len(faces):03d}.jpg")
                    else:
                        flipped_output_path = os.path.join(output_dir, f"{base_name}_flipped.jpg")
                    flipped_face.save(flipped_output_path)
                    logger.info(f"已保存水平翻转图片: {flipped_output_path}")
        
        return cropped_faces
    
    def crop_largest_face_in_image(self, image_path, output_dir=None, crop_size=(512, 512), expand_ratio=1.8, save_flipped=False, filename_prefix="", face_count=0):
        """
        裁切图片中占比最大的人脸
        
        Args:
            image_path (str): 图片路径
            output_dir (str): 输出目录，如果为None则不保存
            crop_size (tuple): 裁切后的图片尺寸 (width, height)
            expand_ratio (float): 扩展比例，控制裁切范围，增大到1.8以包含更多头部区域
            save_flipped (bool): 是否同时保存水平翻转的图片
            filename_prefix (str): 文件名前缀，用于重命名
            face_count (int): 人脸计数，用于全局命名
            
        Returns:
            tuple: (原始裁切图片, 水平翻转图片, 标记人脸的原图) 或 (原始裁切图片, None, 标记人脸的原图) 如果没有检测到人脸则返回 (None, None, None)
        """
        # 检测最大人脸
        largest_face = self.detect_largest_face(image_path)
        
        if largest_face is None:
            logger.info(f"未检测到人脸: {image_path}")
            return None, None, None
        
        # 读取图片
        img = Image.open(image_path)
        
        # 创建标记了人脸位置的图片
        marked_img = self.visualize_faces(image_path)
        
        # 裁切最大人脸
        cropped_face = self.crop_face(img, largest_face, crop_size, expand_ratio)
        
        # 保存原始图片（如果指定了输出目录）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if filename_prefix:
                # 使用指定前缀+数字命名
                output_path = os.path.join(output_dir, f"{filename_prefix}_{face_count + 1:03d}.jpg")
            else:
                # 使用原始文件名
                filename = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{filename}_largest_face.jpg")
                
            cropped_face.save(output_path)
            logger.info(f"已保存原始裁切图片: {output_path}")
            
            # 如果需要保存水平翻转的图片
            flipped_face = None
            if save_flipped:
                flipped_face = cropped_face.transpose(Image.FLIP_LEFT_RIGHT)
                if filename_prefix:
                    # 使用指定前缀+数字命名
                    flipped_output_path = os.path.join(output_dir, f"{filename_prefix}_{face_count + 2:03d}.jpg")
                else:
                    # 使用原始文件名
                    filename = os.path.splitext(os.path.basename(image_path))[0]
                    flipped_output_path = os.path.join(output_dir, f"{filename}_largest_face_flipped.jpg")
                    
                flipped_face.save(flipped_output_path)
                logger.info(f"已保存水平翻转图片: {flipped_output_path}")
            
            return cropped_face, flipped_face, marked_img
        
        return cropped_face, None if not save_flipped else cropped_face.transpose(Image.FLIP_LEFT_RIGHT), marked_img
    
    def visualize_faces(self, image_path, output_path=None, show_largest_only=False):
        """
        在原图上标记检测到的人脸并保存
        
        Args:
            image_path (str): 图片路径
            output_path (str): 输出路径，如果为None则不保存
            show_largest_only (bool): 是否只标记最大人脸
            
        Returns:
            PIL.Image: 标记了人脸位置的图片
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"无法读取图片: {image_path}")
            return None
            
        faces = self.detect_faces(image_path)
        
        # 在图片上绘制人脸框
        if show_largest_only and faces:
            # 只标记最大人脸
            largest_face = None
            largest_area = 0
            
            for (x, y, w, h) in faces:
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, w, h)
            
            if largest_face:
                (x, y, w, h) = largest_face
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        elif faces:
            # 标记所有人脸
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 转换为PIL Image格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(img_rgb)
        
        # 保存图片（如果指定了输出路径）
        if output_path:
            result_img.save(output_path)
            logger.info(f"已保存标记图片: {output_path}")
        
        return result_img


def process_images(src_dir, dst_dir, crop_size, expand_ratio, mark_faces, largest_face_only=False, 
                   min_neighbors=5, scale_factor=1.1, min_size=(30, 30), save_flipped=False, confidence_threshold=0.5,
                   filename_prefix="", use_mtcnn=False, use_yolov8=False, use_blazeface=False, use_ultraface=False,
                   strict_face_filter=True, face_detector=None):
    """
    处理文件夹中的所有图片
    
    Args:
        src_dir (str): 源图片文件夹路径
        dst_dir (str): 输出文件夹路径
        crop_size (tuple): 裁剪尺寸 (width, height)
        expand_ratio (float): 扩展比例
        mark_faces (bool): 是否标记人脸位置
        largest_face_only (bool): 是否只处理最大人脸
        min_neighbors (int): 人脸检测参数，值越大检测越严格
        scale_factor (float): 人脸检测参数，检测窗口缩放因子
        min_size (tuple): 人脸检测参数，最小人脸尺寸
        save_flipped (bool): 是否同时保存水平翻转的图片
        confidence_threshold (float): DNN检测器置信度阈值
        filename_prefix (str): 文件名前缀，用于重命名
        use_mtcnn (bool): 是否使用MTCNN检测器（如果可用）
        use_yolov8 (bool): 是否使用YOLOv8-face检测器（如果可用）
        use_blazeface (bool): 是否使用BlazeFace检测器（如果可用）
        use_ultraface (bool): 是否使用Ultra-Light-Fast-Generic-Face-Detector（如果可用）
        strict_face_filter (bool): 是否使用严格的人脸面积过滤
        face_detector (str): 人脸检测算法名称，可选值: "yolov8", "mtcnn", "blazeface", "ultraface", "opencv_dnn", "haar_cascade"
    """
    # 创建面部裁剪器实例
    cropper = FaceCropper(min_neighbors, scale_factor, min_size, confidence_threshold, 
                          use_mtcnn, use_yolov8, use_blazeface, use_ultraface, strict_face_filter, 
                          face_detector)
    
    # 支持的图片格式
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
    
    # 获取所有图片文件
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(src_dir, extension)))
        image_paths.extend(glob.glob(os.path.join(src_dir, extension.upper())))
    
    # 去除重复的文件路径并排序
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        logger.error(f"在目录 {src_dir} 中未找到图片文件")
        return
    
    input_image_count = len(image_paths)
    logger.info(f"找到 {input_image_count} 个图片文件")
    
    # 创建输出目录
    os.makedirs(dst_dir, exist_ok=True)
    
    # 统计输出的人脸图片数量
    output_face_count = 0
    
    # 处理每个图片文件
    for idx, image_path in enumerate(tqdm(image_paths, desc="处理图片")):
        try:
            # 获取文件名（不含扩展名）
            basename = os.path.splitext(os.path.basename(image_path))[0]
            
            # 标记人脸位置（如果需要）
            if mark_faces:
                marked_output_path = os.path.join(dst_dir, f"{basename}_marked.jpg")
                cropper.visualize_faces(image_path, marked_output_path, show_largest_only=largest_face_only)
            
            # 裁剪人脸
            # 修改输出目录为 image_output/face
            face_output_dir = os.path.join(dst_dir, "face")
            
            # 构造文件名前缀
            if filename_prefix:
                current_prefix = f"{filename_prefix}_{idx+1:04d}"
            else:
                current_prefix = ""
                
            if largest_face_only:
                # 只裁剪最大人脸
                original_face, flipped_face, marked_face = cropper.crop_largest_face_in_image(
                    image_path, 
                    face_output_dir, 
                    crop_size=crop_size,
                    expand_ratio=expand_ratio,
                    save_flipped=save_flipped,
                    filename_prefix=current_prefix
                )
                if original_face:
                    output_face_count += 1
                    if save_flipped and flipped_face:
                        output_face_count += 1
                    logger.info(f"处理完成: {image_path}，裁剪了最大人脸")
                else:
                    logger.info(f"处理完成: {image_path}，未检测到人脸")
            else:
                # 裁剪所有人脸（保持原有功能）
                cropped_faces = cropper.crop_all_faces_in_image(
                    image_path, 
                    face_output_dir, 
                    crop_size=crop_size,
                    expand_ratio=expand_ratio,
                    save_flipped=save_flipped,
                    filename_prefix=current_prefix
                )
                face_count = len(cropped_faces)
                output_face_count += face_count
                if save_flipped:
                    # 每个人脸都有对应的翻转图片
                    output_face_count += face_count
                logger.info(f"处理完成: {image_path}，裁剪了 {face_count} 个人脸")
            
        except Exception as e:
            logger.error(f"处理图片 {image_path} 时出错: {e}")
    
    logger.info(f"任务完成: 输入图片 {input_image_count} 张，输出人脸图片 {output_face_count} 张")


def load_config(config_path):
    """
    从JSON配置文件加载参数
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置参数字典
    """
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"读取配置文件时出错: {e}")
        return None


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从指定文件夹中的图片裁剪人脸",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--src_dir", type=str, required=False, 
                        help="包含源图片的目录 / directory containing source images")
    parser.add_argument("--dst_dir", type=str, required=False, 
                        help="保存裁剪后人脸图片的目录 / directory to save cropped face images")
    parser.add_argument("--crop_size", type=str, default="512,512",
                        help="裁剪尺寸 width,height / crop size 'width,height' (default: 512,512)")
    parser.add_argument("--expand_ratio", type=float, default=1.8,
                        help="裁剪范围扩展比例 / expand ratio for cropping area (default: 1.8)")
    parser.add_argument("--mark_faces", action="store_true",
                        help="在原图上标记检测到的人脸 / mark detected faces on original images")
    parser.add_argument("--config_file", type=str, default="./config/config.json",
                        help="配置文件路径 / path to config file (default: ./config/config.json)")
    parser.add_argument("--largest_face_only", action="store_true",
                        help="只检测和裁剪图片中占比最大的人脸 / detect and crop only the largest face in the image")
    parser.add_argument("--min_neighbors", type=int, default=5,
                        help="人脸检测参数，值越大检测越严格 / face detection parameter, higher value means stricter detection (default: 5)")
    parser.add_argument("--scale_factor", type=float, default=1.1,
                        help="人脸检测参数，检测窗口缩放因子 / face detection parameter, detection window scale factor (default: 1.1)")
    parser.add_argument("--min_size", type=str, default="30,30",
                        help="人脸检测参数，最小人脸尺寸 width,height / face detection parameter, minimum face size 'width,height' (default: 30,30)")
    parser.add_argument("--save_flipped", action="store_true",
                        help="同时保存水平翻转的图片 / also save horizontally flipped images")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="DNN检测器置信度阈值 / DNN detector confidence threshold (default: 0.5)")
    parser.add_argument("--filename_prefix", type=str, default="",
                        help="文件名前缀，用于重命名图片 / filename prefix for renaming images")
    parser.add_argument("--use_mtcnn", action="store_true",
                        help="使用MTCNN检测器（如果可用） / use MTCNN detector (if available)")
    parser.add_argument("--use_yolov8", action="store_true",
                        help="使用YOLOv8-face检测器（如果可用） / use YOLOv8-face detector (if available)")
    parser.add_argument("--use_blazeface", action="store_true",
                        help="使用BlazeFace检测器（如果可用） / use BlazeFace detector (if available)")
    parser.add_argument("--use_ultraface", action="store_true",
                        help="使用Ultra-Light-Fast-Generic-Face-Detector（如果可用） / use Ultra-Light-Fast-Generic-Face-Detector (if available)")
    parser.add_argument("--face_detector", type=str, choices=["yolov8", "mtcnn", "blazeface", "ultraface", "opencv_dnn", "haar_cascade"],
                        help="指定人脸检测算法 / specify face detection algorithm")
    parser.add_argument("--no-strict-filter", action="store_true",
                        help="禁用严格的人脸面积过滤 / disable strict face area filtering")
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    
    # 尝试从配置文件加载参数
    config = load_config(args.config_file)
    
    # 使用配置文件中的参数，如果有的话
    src_dir = config.get("src_dir") if config else None
    dst_dir = config.get("dst_dir") if config else None
    crop_size_str = config.get("crop_size", "512,512") if config else args.crop_size
    expand_ratio = config.get("expand_ratio", 1.8) if config else args.expand_ratio
    mark_faces = config.get("mark_faces", False) if config else args.mark_faces
    largest_face_only = config.get("largest_face_only", False) if config else args.largest_face_only
    min_neighbors = config.get("min_neighbors", 5) if config else args.min_neighbors
    scale_factor = config.get("scale_factor", 1.1) if config else args.scale_factor
    min_size_str = config.get("min_size", "30,30") if config else args.min_size
    save_flipped = config.get("save_flipped", False) if config else args.save_flipped
    confidence_threshold = config.get("confidence_threshold", 0.5) if config else args.confidence_threshold
    filename_prefix = config.get("filename_prefix", "") if config else args.filename_prefix
    use_mtcnn = config.get("use_mtcnn", False) if config else args.use_mtcnn
    use_yolov8 = config.get("use_yolov8", False) if config else args.use_yolov8
    use_blazeface = config.get("use_blazeface", False) if config else args.use_blazeface
    use_ultraface = config.get("use_ultraface", False) if config else args.use_ultraface
    face_detector = config.get("face_detector", None) if config else args.face_detector
    strict_face_filter = config.get("strict_face_filter", True) if config else not args.no_strict_filter
    
    # 命令行参数优先级高于配置文件
    if args.src_dir:
        src_dir = args.src_dir
    if args.dst_dir:
        dst_dir = args.dst_dir
    if args.crop_size != "512,512":  # 默认值
        crop_size_str = args.crop_size
    if args.expand_ratio != 1.8:  # 默认值
        expand_ratio = args.expand_ratio
    if args.mark_faces:
        mark_faces = args.mark_faces
    if args.largest_face_only:
        largest_face_only = args.largest_face_only
    if args.min_neighbors != 5:  # 默认值
        min_neighbors = args.min_neighbors
    if args.scale_factor != 1.1:  # 默认值
        scale_factor = args.scale_factor
    if args.min_size != "30,30":  # 默认值
        min_size_str = args.min_size
    if args.save_flipped:
        save_flipped = args.save_flipped
    if args.confidence_threshold != 0.5:  # 默认值
        confidence_threshold = args.confidence_threshold
    if args.filename_prefix:
        filename_prefix = args.filename_prefix
    if args.use_mtcnn:
        use_mtcnn = args.use_mtcnn
    if args.use_yolov8:
        use_yolov8 = args.use_yolov8
    if args.use_blazeface:
        use_blazeface = args.use_blazeface
    if args.use_ultraface:
        use_ultraface = args.use_ultraface
    if args.face_detector:
        face_detector = args.face_detector
    if args.no_strict_filter:
        strict_face_filter = False
    
    # 检查必需参数
    if not src_dir or not dst_dir:
        logger.error("必须提供源目录和目标目录，可以通过命令行参数或配置文件指定")
        logger.error("通过命令行参数: --src_dir 源目录 --dst_dir 目标目录")
        logger.error("通过配置文件: 在 ./config/config.json 中设置 src_dir 和 dst_dir 参数")
        exit(1)
    
    # 解析裁剪尺寸
    try:
        crop_width, crop_height = map(int, crop_size_str.split(","))
        crop_size = (crop_width, crop_height)
    except ValueError:
        logger.error("crop_size 格式错误，请使用 'width,height' 格式，例如 '512,512'")
        exit(1)
    
    # 解析最小人脸尺寸
    try:
        min_width, min_height = map(int, min_size_str.split(","))
        min_size = (min_width, min_height)
    except ValueError:
        logger.error("min_size 格式错误，请使用 'width,height' 格式，例如 '30,30'")
        exit(1)
    
    # 处理图片
    process_images(
        src_dir=src_dir,
        dst_dir=dst_dir,
        crop_size=crop_size,
        expand_ratio=expand_ratio,
        mark_faces=mark_faces,
        largest_face_only=largest_face_only,
        min_neighbors=min_neighbors,
        scale_factor=scale_factor,
        min_size=min_size,
        save_flipped=save_flipped,
        confidence_threshold=confidence_threshold,
        filename_prefix=filename_prefix,
        use_mtcnn=use_mtcnn,
        use_yolov8=use_yolov8,
        use_blazeface=use_blazeface,
        use_ultraface=use_ultraface,
        strict_face_filter=strict_face_filter,
        face_detector=face_detector
    )