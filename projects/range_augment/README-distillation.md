# Distillation with RangeAugment

We use ResNet-101 trained with RangeAugment as a teacher and trained four mobile models using knowledge distillation. The config files used in our distillation experiments are [here](distillation). Please follow instructions [here](README-classification.md) for training and evaluation on the ImageNet dataset.

## Results on the ImageNet dataset

| Student Model | Top-1 (ERM) | Top-1 (Distillation) | Config                                                                        | Weights                                                                                                                        | Logs                                                                                                                                  |
| --- |----------------------|---------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| MobileNetv1 | 73.8 | 75.2  | [Distil-MV1-config](distillation/teacher_resnet101_student_mobilenet_v1.yaml) | [Distil-MV1-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/distillation/teacher_resnet101_student_mobilenet_v1.pt) | [Distil-MV1-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/distillation/teacher_resnet101_student_mobilenet_v1_logs.txt) |
| MobileNetv2 | 73.0 | 73.4  | [Distil-MV2-config](distillation/teacher_resnet101_student_mobilenet_v2.yaml) | [Distil-MV2-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/distillation/teacher_resnet101_student_mobilenet_v2.pt) | [Distil-MV2-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/distillation/teacher_resnet101_student_mobilenet_v2_logs.txt) |
| MobileNetv3 | 75.1 | 76.0  | [Distil-MV3-config](distillation/teacher_resnet101_student_mobilenet_v3.yaml) | [Distil-MV3-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/distillation/teacher_resnet101_student_mobilenet_v3.pt) | [Distil-MV3-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/distillation/teacher_resnet101_student_mobilenet_v3_logs.txt) |
| MobileViT | 78.2 | 79.4  | [Distil-MViT-config](distillation/teacher_resnet101_student_mobilevit.yaml)   | [Distil-MViT-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/distillation/teacher_resnet101_student_mobilevit.pt)   | [Distil-MViT-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/distillation/teacher_resnet101_student_mobilevit_logs.txt)   |

***Note***:
   * ERM results are from [here](README-classification.md)
   * For MobileViT, we use EMA checkpoint (as suggested in the paper).
