# CelebA 人脸验证实验总结

## 实验目的与设置

本实验比较 Cross-Entropy（CE）、ArcFace 和 Triplet 三种训练目标在未见人物的人脸验证任务上的表现。骨干网络为 VGGFace2 预训练的 InceptionResNetV1，输入尺寸为 160×160，输出 512 维人脸特征。三组实验使用相同的数据路径、manifest 路径和随机种子 42，在 Explorer 的 CUDA 环境完成 15 轮训练。

训练集包含 162,734 张图片、8,156 个身份；验证集包含 19,862 张图片、980 个身份；测试集包含 19,959 张图片、997 个身份。记录显示训练与验证、训练与测试身份交集均为零。每轮验证使用 2,000 对图片，最终测试使用 10,000 对图片。

三组实验的 batch size 均为 64，骨干学习率为 0.0001，weight decay 为 0.0001。CE 和 ArcFace 的分类头学习率为 0.001，第一轮冻结骨干；ArcFace 设置 s=64、m=0.5。Triplet 从第一轮更新骨干，margin=0.5，每批采样 16 个身份、每个身份 4 张图片。因此本实验比较的是这三套配置，不能将所有差异都归因于 loss 本身。

## 最终测试结果

以下均为最佳验证 AUC checkpoint 的测试结果，而非最后一轮模型结果。TPR 表示同一人的图片对被正确接受的比例；FPR 表示不同人的图片对被错误接受的比例。

- CE：ROC AUC **0.98027000**；TPR@FPR=0.1% 为 **40.42%**；TPR@FPR=1% 为 **74.04%**；TPR@FPR=10% 为 **95.08%**。
- ArcFace：ROC AUC **0.98277232**；TPR@FPR=0.1% 为 **85.44%**；TPR@FPR=1% 为 **91.44%**；TPR@FPR=10% 为 **95.84%**。
- Triplet：ROC AUC **0.98290528**；TPR@FPR=0.1% 为 **47.72%**；TPR@FPR=1% 为 **83.64%**；TPR@FPR=10% 为 **96.60%**。

## 训练过程

CE 的训练 loss 从 8.7371 降至 8.0330，最高验证 AUC 为 0.975015，出现在第 2 轮；第 15 轮验证 AUC 为 0.950853。ArcFace 的训练 loss 从 31.9246 降至 5.3233，最高验证 AUC 为 0.983278，同样出现在第 2 轮；第 15 轮为 0.977067。二者后续 loss 下降，但验证 AUC 没有改善，存在泛化退化、与过拟合一致的迹象；仅凭这些日志不能确定具体原因。

Triplet 的训练 loss 从 0.3718 降至 0.2166，最高验证 AUC 为 0.984218，出现在第 14 轮；第 15 轮为 0.981887。不同 loss 的数值尺度不同，不应直接比较三者 loss 大小。

最佳轮次由 metrics.jsonl 中的验证 AUC 和项目 checkpoint 选择逻辑推得：CE 第 2 轮、ArcFace 第 2 轮、Triplet 第 14 轮。本整理未加载 .pt 文件检查其内部 epoch。

## 分析与结论

在本次实验中，ArcFace 在严格限制误认率的条件下表现最好。FPR=1% 时，ArcFace 相比 CE 提高 17.40 个百分点，相比 Triplet 提高 7.80 个百分点；FPR=0.1% 时，分别提高 45.02 和 37.72 个百分点。如果项目的重点是降低不同人物被错认的风险，当前结果支持优先考虑 ArcFace。

Triplet 的 AUC 最高，且在 FPR=10% 时 TPR 最高，但其 AUC 相比 ArcFace 仅高 0.00013296。AUC 概括整条 ROC 曲线，不能替代目标工作点的表现；没有重复实验或置信区间，不能认定这一微小 AUC 差异具有统计显著性。

本实验为身份不重叠的开放集人脸验证，不能报告闭集身份分类 accuracy。summary.json 中 test_accuracy / test_knn_accuracy 为 null 是评估协议导致的正常结果。91.44% 应写成“TPR at 1% FPR”，不能写成“人脸分类准确率 91.44%”。

## 结果范围与后续验证

这些结论来自单一随机种子和 CelebA 数据；尚未报告多次运行的均值或置信区间。未微调基线已完成，见文末补充。10,000 对测试图片仍不足以稳定刻画极低 FPR 的尾部表现。相同配置路径与 seed 支持实验可比性，但本备份没有 manifest 文件或测试图片对清单，不能仅凭路径确认数据和图片对逐项完全一致。

下一步建议增加随机种子重复实验并报告不确定性；若应用以低 FPR 为目标，可另行用验证集相应 TPR 选择 checkpoint。部署阈值应在独立验证集上确定，再在测试集验证；这里报告的 ROC 工作点不是已经校准好的部署阈值。

## 文件与来源

本总结直接核对本地备份 face-reco-results.tar.gz (local checkpoint archive) 内的 config.json、summary.json，以及训练记录。备份包含三组 best.pt、config.json、metrics.jsonl 和 summary.json，共 12 个文件。

- ArcFace：runs/arcface/20261013T185644Z/
- CE：runs/ce/20261013T212022Z/
- Triplet：runs/triplet/20261013T231119Z/

该压缩包是模型与结果备份，不包含数据集、数据划分 manifest 或完整环境依赖快照。运行目录中的 UTC 时间标识保持原样。

## 可直接用于报告的英文结论

We fine-tuned a VGGFace2-pretrained InceptionResNetV1 on CelebA using cross-entropy, ArcFace, and triplet loss under an identity-disjoint verification protocol. All runs used 15 epochs, and checkpoints were selected by validation ROC AUC. On 10,000 test pairs, ArcFace achieved 91.44% TPR at 1% FPR and 85.44% TPR at 0.1% FPR, outperforming CE and triplet at these operating points. Triplet achieved the highest overall ROC AUC (0.982905), narrowly above ArcFace (0.982772), but this difference was not assessed for statistical significance. These results favor ArcFace for low-false-acceptance verification in this experiment, rather than establishing a universal ranking of the losses.

## 未微调基线补充

Explorer 任务 11058609 成功完成，使用相同规模的测试划分、seed=42 和 10,000 对图片。未微调的 VGGFace2 预训练模型达到 AUC 0.98146832，TPR@FPR=0.1% 为 71.64%，TPR@FPR=1% 为 87.32%，TPR@FPR=10% 为 95.44%。

ArcFace 相比基线在 FPR=1% 时提升 4.12 个百分点，在 FPR=0.1% 时提升 13.80 个百分点。CE 和 Triplet 在这两个工作点均低于未微调基线，因此不能笼统声称所有微调方案都有收益。基线结果强化了 ArcFace 在本次低误认率评估中的优势，但仍需多种子重复实验确认稳定性。基线记录来自用户粘贴的远程结果，三组历史微调任务没有保存 manifest 校验值，不能回溯证明四次运行使用的文件内容完全一致。
