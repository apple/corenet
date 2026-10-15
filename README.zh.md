# CoreNet：用于训练深度神经网络的高性能库

CoreNet 是一个深度神经网络算法套件，赋能研究人员与算法工程师训练标准以及新颖的轻量化与超大规模模型，适用于多种复杂任务，包括基础大模型（例如 CLIP 与大语言模型 LLM）、目标分类、目标检测以及语义分割。

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 目录

   * [最新动态](#最新动态)
   * [Apple 基于 CoreNet 开展的研究工作](#apple-基于-corenet-开展的研究工作)
   * [安装指南](#安装指南)
   * [目录架构详解](#目录架构详解)
   * [项目维护者](#项目维护者)
   * [参与贡献](#参与贡献)
   * [开源许可证](#开源许可证)
   * [与 CVNets 的渊源与演进](#与-cvnets-的渊源与演进)
   * [论文引用](#论文引用)

## 最新动态

   * ***2024年10月***：CoreNet 库 0.1.1 版本发布，包含：
      * [KV Prediction (首字时延优化 / KV 缓存预测)](./projects/kv-prediction/)

## Apple 基于 CoreNet 开展的研究工作

以下是 Apple 使用 CoreNet 发表的学术研究论文列表。此外，训练与评估配方以及预训练模型权重链接均可在 [projects](./projects/) 目录中找到，详情请参阅各项目子目录。

   * [KV Prediction for Improved Time to First Token](https://arxiv.org/abs/2410.08391)（KV 预测加速首 Token 生成时延）
   * [OpenELM: An Efficient Language Model Family with Open Training and Inference Framework](https://arxiv.org/abs/2404.14619)（OpenELM：具备完全开源训练与推理框架的高效大语言模型家族）
   * [CatLIP: CLIP-level Visual Recognition Accuracy with 2.7x Faster Pre-training on Web-scale Image-Text Data](https://arxiv.org/abs/2404.15653)（CatLIP：在网络级图文数据上以 2.7 倍更快预训练速度达到 CLIP 级视觉识别精度）
   * [Reinforce Data, Multiply Impact: Improved Model Accuracy and Robustness with Dataset Reinforcement](https://arxiv.org/abs/2303.08983)（强化数据，成倍放大影响力：通过数据集强化提升模型精度与鲁棒性）
   * [CLIP meets Model Zoo Experts: Pseudo-Supervision for Visual Enhancement](https://arxiv.org/abs/2310.14108)（CLIP 与模型库专家结合：用于视觉增强的伪监督学习）
   * [FastVit: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/abs/2303.14189)（FastVit：采用结构重参数化的高速混合视觉 Transformer）
   * [Bytes Are All You Need: Transformers Operating Directly on File Bytes](https://arxiv.org/abs/2306.00238)（字节即一切：直接基于文件原始字节运行的 Transformer）
   * [MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040)（MobileOne：优化的 1 毫秒级端侧移动骨干网络）
   * [RangeAugment: Efficient Online Augmentation with Range Learning](https://arxiv.org/abs/2212.10553)（RangeAugment：基于范围学习的高效在线数据增强）
   * [Separable Self-attention for Mobile Vision Transformers (MobileViTv2)](https://arxiv.org/abs/2206.02680)（适用于移动视觉 Transformer 的可分离自注意力机制 MobileViTv2）
   * [CVNets: High performance library for Computer Vision, ACM MM'22](https://arxiv.org/abs/2206.02002)（CVNets：面向计算机视觉的高性能库）
   * [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer, ICLR'22](https://arxiv.org/abs/2110.02178)（MobileViT：轻量级、通用且移动端友好的视觉 Transformer）

## 安装指南

运行测试、使用 Jupyter Notebook（[安装指引](https://jupyter.org/install)）以及参与代码贡献均需要安装 Git LFS（配置说明见下方），因此建议您首先安装并启用 Git LFS。

在 Linux 平台上，推荐使用 Python 3.10+ 与 PyTorch（版本 >= v2.1.0）；在 macOS 平台上，系统自带的 Python 3.9+ 即可满足需求。

请注意：若您希望参与代码贡献或运行测试套件，则必须安装下方列出的可选依赖。

**针对 Linux 系统（请将 `apt` 替换为您发行版的包管理器）：**

```bash
sudo apt install git-lfs

git clone git@github.com:apple/corenet.git
cd corenet
git lfs install
git lfs pull
# 以下 venv 命令为可选但强烈推荐。或者，您也可以创建并激活 conda 环境。
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --editable .
```

安装音频与视频处理的可选依赖：

```bash
sudo apt install libsox-dev ffmpeg
```

**针对 macOS 系统（假设您使用 Homebrew）：**

```bash
brew install git-lfs

git clone git@github.com:apple/corenet.git
cd corenet
cd $(pwd -P)  # 详见下方注意事项。
git lfs install
git lfs pull
# 以下 venv 命令为可选但强烈推荐。或者，您也可以创建并激活 conda 环境。
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --editable .
```

安装音频与视频处理的可选依赖：

```bash
brew install sox ffmpeg
```

> **注意**：在 macOS 上，文件系统默认对大小写不敏感，而大小写问题可能导致 Git 出现异常。您应当以大小写敏感的方式访问磁盘上的仓库，即路径的大小写必须与 `ls` 命令列出的目录名严格保持一致。您可以使用 `cd $(pwd -P)` 命令快速切换到该物理规范路径。

## 目录架构详解

本章节提供了 CoreNet 关键核心目录的快速访问入口与简要说明：

<table>
<thead>
<tr>
<th> 描述 </th>
<th> 快速访问 </th>
</tr>
</thead>
<tbody>
<tr> <td> <h3> 新手上路 (Getting Started) </h3>
通过运行官方示例是快速上手 CoreNet 的最佳途径。
</td> <td> <pre>
└── tutorials
    ├── <a href="tutorials/train_a_new_model_on_a_new_dataset_from_scratch.ipynb">train_a_new_model_on_a_new_dataset_from_scratch.ipynb</a>
    ├── <a href="tutorials/guide_slurm_and_multi_node_training.md">guide_slurm_and_multi_node_training.md</a>
    ├── <a href="tutorials/clip.ipynb">clip.ipynb</a>
    ├── <a href="tutorials/semantic_segmentation.ipynb">semantic_segmentation.ipynb</a>
    └── <a href="tutorials/object_detection.ipynb">object_detection.ipynb</a>
</pre> </td> </tr>

<tr> <td> <h3> 训练配方 (Training Recipes) </h3>
CoreNet 提供了可完全复现的训练配方，并在 <code>projects/</code> 目录下提供了已发表论文的预训练权重与检查点。

各项目论文目录通常包含以下内容：

* `README.md`：提供文档说明、预训练权重下载链接与引用信息。
* `<task_name>/<model_name>.yaml`：提供用于复现模型训练与评测的完整配置文件。
</td> <td> <pre>
└── projects
    ├── <a href="projects/kv-prediction">kv-prediction</a> (*)
    ├── <a href="projects/byteformer">byteformer</a>
    ├── <a href="projects/catlip">catlip</a>
    ├── <a href="projects/clip">clip</a>
    ├── <a href="projects/fastvit">fastvit</a>
    ├── <a href="projects/mobilenet_v1">mobilenet_v1</a>
    ├── <a href="projects/mobilenet_v2">mobilenet_v2</a>
    ├── <a href="projects/mobilenet_v3">mobilenet_v3</a>
    ├── <a href="projects/mobileone">mobileone</a>
    ├── <a href="projects/mobilevit">mobilevit</a>
    ├── <a href="projects/mobilevit_v2">mobilevit_v2</a>
    ├── <a href="projects/openelm">openelm</a>
    ├── <a href="projects/range_augment">range_augment</a>
    ├── <a href="projects/resnet">resnet</a>
    └── <a href="projects/vit">vit</a>
<br>
(*) 新发布项目。
</pre> </td> </tr>

<tr> <td> <h3> MLX 示例 </h3>
MLX 示例演示了如何在 Apple Silicon 芯片上极致高效地运行 CoreNet 模型。
更多详细信息请参阅对应示例目录下的 <code>README.md</code> 文件。

</td> <td> <pre>
└──mlx_example
    ├── <a href="mlx_examples/clip">clip</a>
    └── <a href="mlx_examples/open_elm">open_elm</a>
</pre> </td> </tr>

<tr> <td> <h3> 模型实现 (Model Implementations) </h3>
模型按任务类型进行分类管理（例如 "classification" 图像分类）。您可以在对应的任务子文件夹中找到该任务下的所有模型实现。

每个模型类均由装饰器
<code>@MODEL_REGISTRY.register(name="&lt;model_name&gt;", type="&lt;task_name&gt;")</code> 注册。
如需在 CoreNet 训练或评估中使用某个模型类，只需在 YAML 配置文件中指定 <code>models.&lt;task_name&gt;.name = &lt;model_name&gt;</code> 即可。

</td> <td> <pre>
└── corenet
    └── modeling
        └── <a href="corenet/modeling/models">models</a>
            ├── <a href="corenet/modeling/models/audio_classification">audio_classification</a>
            ├── <a href="corenet/modeling/models/classification">classification</a>
            ├── <a href="corenet/modeling/models/detection">detection</a>
            ├── <a href="corenet/modeling/models/language_modeling">language_modeling</a>
            ├── <a href="corenet/modeling/models/multi_modal_img_text">multi_modal_img_text</a>
            └── <a href="corenet/modeling/models/segmentation">segmentation</a>
</pre> </td> </tr>

<tr> <td> <h3> 数据集 (Datasets) </h3>
与模型类似，数据集同样按任务类型进行组织与分类。
</td> <td> <pre>
└── corenet
    └── data
        └── <a href="corenet/data/datasets">datasets</a>
            ├── <a href="corenet/data/datasets/audio_classification">audio_classification</a>
            ├── <a href="corenet/data/datasets/classification">classification</a>
            ├── <a href="corenet/data/datasets/detection">detection</a>
            ├── <a href="corenet/data/datasets/language_modeling">language_modeling</a>
            ├── <a href="corenet/data/datasets/multi_modal_img_text">multi_modal_img_text</a>
            └── <a href="corenet/data/datasets/segmentation">segmentation</a>
</pre> </td> </tr>

<tr> <td> <h3> 其他核心模块目录 </h3>
在此部分中，我们重点展示了实现 YAML 配置文件所引用类的其他关键目录。
</td> <td> <pre>
└── corenet
    ├── <a href="corenet/loss_fn">loss_fn</a>（损失函数）
    ├── <a href="corenet/metrics">metrics</a>（评估指标）
    ├── <a href="corenet/optims">optims</a>（优化器）
    │   └── <a href="corenet/optims/scheduler">scheduler</a>（学习率调度器）
    ├── <a href="corenet/train_eval_pipelines">train_eval_pipelines</a>（训练与评估管线）
    ├── <a href="corenet/data">data</a>（数据模块）
    │   ├── <a href="corenet/data/collate_fns">collate_fns</a>（批处理整理函数）
    │   ├── <a href="corenet/data/sampler">sampler</a>（采样器）
    │   ├── <a href="corenet/data/text_tokenizer">text_tokenizer</a>（分词器）
    │   ├── <a href="corenet/data/transforms">transforms</a>（数据预处理与增强）
    │   └── <a href="corenet/data/video_reader">video_reader</a>（视频读取器）
    └── <a href="corenet/modeling">modeling</a>（模型结构）
        ├── <a href="corenet/modeling/layers">layers</a>（基础神经网络层）
        ├── <a href="corenet/modeling/modules">modules</a>（复合组件与算子）
        ├── <a href="corenet/modeling/neural_augmentor">neural_augmentor</a>（神经数据增强器）
        └── <a href="corenet/modeling/text_encoders">text_encoders</a>（文本编码器）
</pre> </td> </tr>

</tbody>
</table>

## 项目维护者

本项目由 <a href="https://mchorton.com" target="_blank">Maxwell Horton</a>、<a href="https://www.mohammad.pro" target="_blank">Mohammad Sekhavat</a>、Yanzi Jin 以及 <a href="https://huggingface.co/depthwise" target="_blank">Dmitry Belenko</a> 共同开发并维护。

### 历任维护者
* <a href="https://sacmehta.github.io" target="_blank">Sachin Mehta</a>
* <a href="https://farzadab.github.io" target="_blank">Farzad Abdolhosseini</a>

## 参与贡献

热烈欢迎开源社区提交 PR 参与贡献！有关如何为 CoreNet 贡献代码的详细信息，请参阅我们的[贡献指南 (CONTRIBUTING.md)](CONTRIBUTING.md)。

请务必遵守我们的 [行为准则 (Code of Conduct)](https://github.com/apple/.github/blob/main/CODE_OF_CONDUCT.md)。

## 开源许可证

开源许可协议详见 [LICENSE](LICENSE)。

## 与 CVNets 的渊源与演进

CoreNet 由 CVNets 演进发展而来，旨在覆盖超越传统计算机视觉的更广泛应用领域。该扩展全面支持了包括大语言模型（LLM）与多模态在内的现代基础大模型的训练。

## 论文引用

如果您发现我们的工作对您的研究有所帮助，请引用以下论文：

```bibtex
@inproceedings{mehta2022cvnets,
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad},
     title = {CVNets: High Performance Library for Computer Vision},
     year = {2022},
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
     series = {MM '22}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年10月15日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
