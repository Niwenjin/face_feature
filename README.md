# face_feature

## 评估指标

评估模型性能的基本指标：

-   TP: True Positive 预测结果为正样本，实际也为正样本
-   FP: False Positive 预测结果为正样本，实际为负样本（误检）
-   TN: True Negative 预测结果为负样本，实际为负样本
-   FN: False Negative 预测结果为负样本，实际为正样本（漏检）

$Precision = \frac{TP}{TP + FP}$

$Recall/TPR = \frac{TP}{TP + FN}$

$FPR = \frac{FP}{FP + TN}$

ROC 曲线：以 FPR 为 x 轴，TPR 为 y 轴。ROC 曲线下的面积（AUC）是衡量模型整体性能的一个指标，AUC 值越高，表示模型的分类性能越好。

PR 曲线：以 Recall 为 x 轴，Precision 为 y 轴。通常通过平衡点（BEP）判断模型的性能，即 P=R 时的取值，如果这个值较大，说明模型的性能较好。PR 曲线下的面积（AUC）也可以作为判断模型性能的指标。

F1 曲线：F1 分数是精确率和召回率的调和平均数，计算公式为 $F1=2×\frac{Precision×Recall}{Precision+Recall}$ 。以阈值为 x 轴，F1 分数为 y 轴，体现了模型在不同阈值下的综合性能。

## 在 IJB 数据集上评估模型

IJB（IARPA Janus Benchmark）数据集是一个由美国国家标准化研究院（NIST）发布的人脸识别数据集，它包含多个子集，其中 IJB-A、IJB-B 和 IJB-C 是最常用的三个部分。

下载 IJB 数据集: [GDrive](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg)

更新元数据 (1:1 and 1:N): [Baidu Cloud](https://pan.baidu.com/s/1x-ytzg4zkCTOTtklUgAhfg) (code:7g8o) ; [GDrive](https://drive.google.com/file/d/1MXzrU_zUESSx_242pRUnVvW_wDzfU8Ky/view?usp=sharing)

```sh
└── IJB_release
    ├── IJBB
    │   ├── loose_crop  # 图像
    │   ├── meta        # 元数据
    │   └── result
    └── IJBC
        ├── loose_crop  # 图像
        ├── meta        # 元数据
        └── result
```

测试分为两阶段进行，如果绘制曲线过程中出错，可以读取保存的 embedding 文件继续进行测试：

1. 在模型对应的推理环境下执行推理，将图像提取得到的 embedding 存储为 npz 格式。

    ```sh
    python ijb_evals.py -m b3/model/feature_opencv_net_rk3588.rknn -d IJB_release -s IJBB -E
    ```

2. 读取 npz 向量文件，执行评估代码，计算各种指标，绘制曲线。

    ```sh
    python ijb_evals.py -R IJB_result/feature_opencv_net_rk3588_IJBB_11.npz -d IJB_release -s IJBB
    ```
