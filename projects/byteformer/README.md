# Bytes Are All You Need: Transformers Operating Directly On File Bytes

[ByteFormer](https://arxiv.org/abs/2306.00238) is a Transformer architecture able to perform inference directly on file bytes. It achieves accuracy greater than traditional image transformers at similar parameter settings (77.33% on ImageNet, compared to 72.2% for the original [DeiT-Ti](https://arxiv.org/pdf/2012.12877.pdf)), without the need for any inference-time image decoding. As our network only consumes bytes, we experiment with different image encodings and different modalities. Without modifying architecture or training hyperparameters, it can achieve competitive performance on Speech Commands v2 (95.42%, compared to state-of-the-art accuracy of 98.7%). We also experiment with obfuscated inputs to our network, to enhance privacy at inference time. See our paper for details.

<p align="center">
<img src="model_arch.png" width="50%" align="center">
</p>

## Training and Evaluation

Training occurred on single node machines with 8 A100 GPUs (for ImageNet) or 4 A100 GPUs (for Speech Commands V2). Training can be done with the configs in subdirectories of the folder in which this README is found. Training uses the following command:

```corenet-train --common.config-file $CONFIG_FILE```

Evaluation can be done using the below example command:

```bash
export CFG_FILE=projects/byteformer/imagenet_file_encodings/encoding_type=TIFF.yaml
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_tiff.pt
export DATASET_PATH=/mnt/vision_datasets/imagenet/validation/

CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```


Each subdirectory corresponds to the experiments in a different table in our paper.

- `imagenet_file_encodings/` contains experiments using TIFF or other encodings.
- `imagenet_jpeg_q100/` contains experiments using JPEG quality factor of 100.
- `imagenet_jpeg_q60/` contains experiments using JPEG quality factor of 60.
- `imagenet_jpeg_shuffle_bytes/` contains ablations that shuffle the byte order.
- `imagenet_obfuscation/` contains experiments swapping byte values (see paper for details).
- `imagenet_privacy_preserving_camera/` contains experiments that mask pixel values.
- `speech_commands_mp3/` contains audio classification experiments on Speech Commands v2 using MP3 files.
- `speech_commands_wav/` contains audio classification experiments on Speech Commands v2 using WAV files.

***Note on reproducing FLOPs/Model Size Estimates***: By default, the embedding sizes (set by `--model.classification.byteformer.max-num-tokens`) set in the configs are larger than necessary, to allow experimentation without having to change the value (e.g. if the kernel size of the Conv1D is lowered, which would result in longer token lengths input to the Transformer backbone). When estimating performance in our paper, we set this value to the average token length (after BF-Ti's Conv1D downsampling) for the given input type (TIFF images, JPEG images, etc.). However, if the values are set to the average input length during training, an error will occur in the case of variable-length inputs (such as JPEG), because an input longer than the `max-num-tokens` will inevitably occur.  Similarly, `--model.classification.byteformer.dummy-input-token-length` should be set to your expected input length (before BF-Ti's Conv1D downsampling) for your particular input domain, for accurate performance estimates.

## Pre-Trained Models

| Dataset                              | Task                 | Top-1 | Config                                                                                               | Weights                                                                                                                                                                                | 
|--------------------------------------|----------------------|-------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ImageNet             | Image Classification | 77.05 | [IN TIFF](imagenet_file_encodings/encoding_type=TIFF.yaml)                                      | [IN TIFF](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_tiff.pt)                                              | 
| ImageNet             | Image Classification | 67.64 | [IN JPEG Q100 k=8 w=128](imagenet_jpeg_q100/conv_kernel_size=8.yaml)                                 | [IN JPEG Q100 k=8 w=128](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_jpeg_q100_k8_w128.pt)                                 | 
| ImageNet             | Image Classification | 62.43 | [IN JPEG Q60 k=4 w=128](imagenet_jpeg_q60/conv_kernel_size=4,window_sizes=[128].yaml)                | [IN JPEG Q60 k=4 w=128](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_jpeg_q60_k4_w128.pt)                                   | 
| ImageNet (Shuffle Bytes)             | Image Classification | 61.14 | [IN Shuffle Bytes Reverse](imagenet_jpeg_shuffle_bytes/mode=reverse.yaml)                            | [IN Shuffle Bytes Reverse](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_shuffle_bytes_reverse.pt) | 
| ImageNet (Obfuscation)               | Image Classification | 76.00 | [IN Random Uniform [-20, 20]](imagenet_obfuscation/width_range=[-20,20].yaml)                        | [IN Random Uniform [-20, 20]](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_obfuscation.pt)                                  | 
| ImageNet (Privacy Preserving Camera) | Image Classification | 68.10 | [IN k=4 keep_pixels=0.05](imagenet_privacy_preserving_camera/keep_frac=0.05,conv_kernel_size=4.yaml) | [IN k=4 keep_pixels=0.05](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/imagenet_privacy_preserving_camera_k4_f0.05.pt)               |
| Speech Commands v2 12-Way            | Audio Classification | 94.95 | [SC WAV FP32 k=32 w=128](speech_commands_wav/encoding_dtype=float32,conv_kernel_size=32.yaml)        | [SC WAV FP32 k=32 w=128](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/speech_commands_wav.pt)                                        | 
| Speech Commands v2 12-Way            | Audio Classification | 90.25 | [SC MP3 k=4 w=32](speech_commands_mp3/conv_kernel_size=4,window_size=[32].yaml)                      | [SC MP3 k=4 w=32](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/multimodal_classification/speech_commands_mp3_k4_w32.pt)                                        | 

***Note on model sizes***: As described in the above note on reproducing estimates of FLOPs and model sizes, the above checkpoints include larger embeddings than necessary to facilitate easier experimentation without repeated config changes. See the above note for more details.

## Citation

If you find our work useful, please cite:

```BibTeX
@article{Horton2023BytesAA,
  title={Bytes Are All You Need: Transformers Operating Directly On File Bytes},
  author={Maxwell Horton and Sachin Mehta and Ali Farhadi and Mohammad Rastegari},
  journal={ArXiv},
  year={2023},
  volume={abs/2306.00238}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```
