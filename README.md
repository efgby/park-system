# Parking System

A modular parking lot vehicle counting framework built for official PyTorch RT-DETR, ByteTrack, and OpenCV.

## Purpose

This project provides a modular baseline for counting vehicles entering and leaving three parking zones:

- A: Front
- B: Left
- C: Right

It is designed so that video input, detection, tracking, counting, and visualization can be replaced independently.

## Project Structure

- `main.py`: application entry point
- `config.yaml`: all runtime parameters and default zone settings
- `detector/`: RT-DETR detector wrapper
- `tracker/`: ByteTrack wrapper
- `counting/`: zone counting logic
- `video/`: MP4 and future RTSP input abstraction
- `visualization/`: OpenCV drawing utilities
- `utils/`: geometry helpers
- `assets/test.mp4`: default test video input
- `weights/`: model weights directory
- `output/`: annotated output video directory

## Environment

Recommended:

- Python 3.10
- Conda environment named `parking-rtdetr`
- PyTorch
- OpenCV
- NumPy
- PyYAML

CPU-only setup for this project:

```bash
conda create -n parking-rtdetr python=3.10 -y
conda activate parking-rtdetr
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt
```

If the official RT-DETR package needs to be installed separately, use the GitHub source listed in `requirements.txt`.

## Installation

```bash
pip install -r requirements.txt
```

For CPU-only execution, install the CPU wheels for PyTorch first, then install the remaining requirements.

## RT-DETR Weights

Place the official RT-DETR weights in:

- `weights/model.pth`

Update `config.yaml` if your local path differs.

## Test Video

Place the test video in:

- `assets/test.mp4`

## Configuration

Edit `config.yaml` to change:

- parking capacities
- virtual line coordinates
- detection confidence threshold
- tracking thresholds
- video source

## Run

```bash
python main.py
```

Press `q` to exit.

## Switching to RTSP

Change `video.type` and `video.source` in `config.yaml` from file input to your RTSP stream. The detector, tracker, and counter modules do not need to be rewritten.

## ENTER / EXIT

The zone counter is designed to detect line crossing events and update occupied counts when a vehicle enters or exits a zone. The current scaffold includes the module boundaries and state storage needed for that logic.

## Known Limits

- RT-DETR inference is not wired yet.
- ByteTrack association is not wired yet.
- Zone crossing logic is scaffolded and will need the final event rules.
- The default test video is not included.
- The default weights file is not included.


我赞同你的想法。

对于校招和实习简历，不要把技术栈写死在具体框架上，因为面试官更关注的是：

你解决了什么问题
掌握了什么能力
用了什么技术路线


而不是：

CatBoost
YOLO
RT-DETR


这些具体模型反而适合面试时展开讲。

简历里可以写得更抽象、更有层次。

工业设备能耗预测与智能运维分析平台

机器学习｜异常检测｜Agent Workflow｜数据分析平台

面向工业设备运行过程中异常发现滞后、分析依赖人工经验的问题，构建设备能耗预测与智能诊断平台。基于回归预测模型建立设备正常工况基线，通过预测偏差分析实现异常检测与健康状态评估；设计关键异常事件识别机制，实现风险预警与运行状态监测；搭建智能分析工作流，将设备运行数据与大语言模型结合，实现异常原因分析、风险评估及运维建议自动生成。

工业视觉缺陷检测系统

计算机视觉｜目标检测｜深度学习｜质量检测

针对传统人工质检效率低、一致性差的问题，开发工业产品缺陷自动检测系统。围绕小目标缺陷、样本不均衡及复杂背景场景开展数据处理和模型优化，实现产品表面缺陷的自动识别与分类；构建检测结果统计与可视化模块，支持质量分析、缺陷追溯及检测报告自动生成。

智能车辆检测与轨迹分析系统

目标检测｜多目标跟踪｜轨迹分析｜视频理解

面向停车场及园区监控场景，构建车辆检测、跟踪与行为分析一体化系统。通过目标检测与多目标跟踪技术实现车辆跨帧身份关联及运动轨迹生成，结合轨迹信息完成方向识别、区域计数与交通流量统计；设计区域状态管理机制，为越界检测和行为分析提供基础支撑，实现从目标识别到行为理解的完整分析流程。

多尺度图神经网络关系预测模型（SCI）

图神经网络｜表示学习｜深度学习

针对复杂网络关系预测中高阶关联特征难以建模的问题，设计多尺度图表示学习模型，融合局部结构信息与全局拓扑特征提升关系建模能力；通过注意力机制增强关键特征表达，并完成模型训练、实验验证及对比分析，相关研究成果发表于SCI期刊。

我觉得你的简历最终最好形成这样的能力画像：

工业AI智能分析平台
↓
工业视觉缺陷检测
↓
视频分析与多目标跟踪
↓
图神经网络科研成果


这样面试官看到的是：

机器学习 + Agent + 工业AI + 计算机视觉 + 科研能力

而不是一堆具体框架名称堆砌。这样的描述通常比直接写“CatBoost、RT-DETR、YOLO”更高级，也更适合简历筛选阶段。


同意，简历不是项目报告。

对于校招/实习，一般控制在 3-4行，突出：

业务问题 → 技术方案 → 你的贡献 → 结果


避免大段背景介绍。

工业设备能耗预测与AI智能运维平台

Python｜CatBoost｜Flask｜n8n｜LLM Agent｜Docker

面向工业设备能耗异常监测场景，构建基于 CatBoost 的功耗预测模型，通过实际功耗与预测功耗残差实现异常检测与健康度评估；设计 Health Score 与 Critical Event 分析机制，实现设备异常识别与风险预警；基于 Flask 搭建可视化平台，并通过 n8n 集成 LLM Agent 实现异常原因分析、风险评估及运维建议自动生成。

工业视觉缺陷检测系统

Python｜OpenCV｜PyTorch｜YOLO/RT-DETR

针对人工质检效率低、漏检率高的问题，构建工业产品缺陷自动检测系统；完成数据标注、数据增强及模型训练优化，提升小目标缺陷识别能力；结合检测结果统计与可视化模块，实现缺陷检测、结果追溯及质量分析自动化。

基于RT-DETR与OC-SORT的车辆检测与轨迹分析系统

Python｜PyTorch｜RT-DETR｜OC-SORT｜OpenCV

构建车辆检测、目标跟踪与轨迹分析一体化系统，实现车辆跨帧身份关联与运动轨迹生成；对比评估 ByteTrack 与 OC-SORT 跟踪性能，优化 ID 连续性与轨迹稳定性；基于轨迹信息实现方向识别、区域计数及交通流量统计，并支持实时可视化展示。

多尺度图神经网络关系预测模型（SCI）

Python｜PyTorch Geometric｜GNN｜Attention｜Contrastive Learning

针对复杂图结构关系预测问题，设计融合局部结构与全局拓扑特征的多尺度图神经网络模型；引入注意力机制与对比学习提升节点表示能力和模型泛化性能；完成模型设计、实验验证及消融分析，相关成果发表于 SCI 期刊。

如果只留 3个项目：

工业设备能耗预测与AI智能运维平台 ⭐⭐⭐⭐⭐
工业视觉缺陷检测系统 ⭐⭐⭐⭐⭐
多尺度图神经网络关系预测（SCI）⭐⭐⭐⭐

如果留 4个项目：

工业设备能耗预测与AI智能运维平台
工业视觉缺陷检测系统
RT-DETR与OC-SORT车辆检测跟踪系统
多尺度图神经网络关系预测（SCI）

这是目前对你找 AI算法工程师 / 机器学习工程师 / 工业AI工程师 最有竞争力的组合。