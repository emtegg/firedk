import streamlit as st
import requests
# from streamlit_lottie import st_lottie
import shutil
import os
import sys
from pathlib import Path
import cv2
import torch

import os.path as osp
# 将项目根目录添加到 sys.path
sys.path.append(str(Path(__file__).resolve().parent))

# 导入YOLOv5特定模块
from models.common import DetectMultiBackend   # 模型加载和推理相关的类
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams   # 用于加载图像和视频流的数据加载器
from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
FILE = Path(__file__).resolve()  # 获取当前脚本文件的绝对路径。
ROOT = FILE.parents[0]           # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))   # 将ROOT添加到PATH中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径



def model_load(weights="", device='', half=False, dnn=False):
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    print("模型加载完成!")
    return model

def upload_img(source):
    if source:
        suffix = source.split(".")[-1]
        save_path = osp.join("images/tmp", "tmp_upload." + suffix)
        shutil.copy(source, save_path)
        return save_path
    else:
        print("请提供有效的图像路径")
        return None
def detect_img(model, source, output_path):
    print(source)
    output_size = 480
    imgsz = [640, 640]  # 推理尺寸
    conf_thres = 0.25  # 置信度阈值
    iou_thres = 0.45  # NMS IOU阈值
    max_det = 1000  # 每张图像的最大检测数量
    device = 'cpu'  # 使用的设备（CPU或GPU）
    view_img = False  # 显示结果
    save_txt = False  # 保存结果到文本文件
    save_conf = False  # 在标签中保存置信度
    save_crop = False  # 保存裁剪的预测框
    nosave = False  # 不保存图像/视频
    classes = None  # 按类别过滤
    agnostic_nms = False  # 类别无关的NMS
    augment = False  # 使用增强推理
    visualize = False  # 可视化特征
    line_thickness = 3  # 边界框厚度
    hide_labels = False  # 隐藏标签
    hide_conf = False  # 隐藏置信度
    half = False  # 使用FP16半精度推理
    dnn = False  # 使用OpenCV DNN进行ONNX推理

    device = select_device(device)
    webcam = False
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸
    save_img = not nosave and not source.endswith('.txt')  # 保存推理图像

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # 批量大小

    vid_path, vid_writer = [None] * bs, [None] * bs

    # 推理
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # 预热

    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8转换为fp16/32
        im /= 255  # 0 - 255转换为0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # 扩展为批量维度
        t2 = time_sync()
        dt[0] += t2 - t1

        # 推理
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # 非极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # 处理预测结果
        for i, det in enumerate(pred):
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # 转换为路径对象
            s += '%gx%g ' % im.shape[2:]  # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益
            imc = im0.copy() if save_crop else im0  # 用于保存裁剪的图像
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # 将边界框从图像尺寸缩放到原始图像尺寸
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类别的检测数
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化的xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式

                    if save_img or save_crop or view_img:
                        c = int(cls)  # 类别整数
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # 打印时间（仅推理）
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # 保存结果（带检测框的图像）
            im0 = annotator.result()
            resize_scale = output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)

            cv2.imwrite(output_path, im0)
            print(f"检测结果已保存至 {output_path}")
            return im0

def main():
    st.title("火 灾 检 测 made by czh")
    st.markdown("22电41机器学习项目实训")

    img_path = st.file_uploader(label="请输入一张图片", type=['jpg', 'png', ])
    st.markdown("### 用户上传图片，显示如下: ")
    st.image(img_path, channels="RGB")

    weights = "./weighs/best.pt"
    source =  img_path
    output_path = './result/1.jpeg'
    device = 'cpu'

    model = model_load(weights=weights, device=device)
    upload_path = upload_img(source)
    print('111111111111111')
    print(upload_path)
    print('111111111111111')
    if upload_path:
        res_im = detect_img(model, upload_path, output_path)


    st.markdown("**请点击按钮开始预测**")
    predict = st.button("类别预测")
    if predict:
        st.image(output_path, channels="RGB")

if __name__ == '__main__':
    main()



