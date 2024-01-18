# 关于REACT2024
**首先，挑战赛官网如下：[REACT2024](https://sites.google.com/cam.ac.uk/react2024/home)**  
这个挑战赛的任务是：建立一个机器学习模型，在双人交互的背景下，通过说话者的视频、音频、表情等数据，生成听者的面部反应并要保证反应的合理性(FRDist and FRCorr)、多样性(FRVar, FRDiv, and FRDvs)、同步性(FRSyn)、真实性(FRRea)。

------------

## DATASET
- **NoXI**: NOvice eXpert Interaction dataset
- **RECOLA**: REmote COLlaborative and Affective dataset  
这两个数据集都是需要找官方并签协议书的，然后官方会发送将两数据集混合筛选好的数据集。（缺点是OneDrive下载数据集会丢失文件）

数据集主要分为以下四部分：
- **cropped_face** (“Speaker-Listener”视频， mp4格式)
- **3D_FV_files** (“Speaker-Listener”3DMM系数， npy格式)
- **Audio_files** (“Speaker-Listener”音频，wav格式)
- **Emotion** (“Speaker-Listener”面部数据，15个面部动作单元(AU)+2个面部影响参数+8个面部表情(中性、开心、悲伤等), csv格式)

为便于网络读取，加快训练速度，我将mp4数据打包成npy数据[seq_len, img_size[0], img_size[1]]，其他不变。

------------

## Challenge Tasks
### 1、Offline Multiple Appropriate Facial Reaction Generation (Offline MAFRG)
输入：说话者的audio_clips和video_cilps, 实际seq_len=751，训练时随机从中裁切256帧作为patch
输出：listener_emotion, listener_3dmm, distribution (latend_dim的分布，这个涉及VAE，暂时对其还不是很了解)

### 2、Online Multiple Appropriate Facial Reaction Generation (Online MAFRG)
输入：说话者的audio_clips和video_cilps，实际seq_len=751，训练时随机从中裁切256帧作为patch
输出：listener_emotion, listener_3dmm, distribution

------------

## Metrics
- Facial reaction correlation (FRCorr)面部反应相关性
- Facial reaction distance (FRDist)面部反应距离
- Appropriate facial reaction prediction accuracy (ACC)合适面部反应预测准确度
- Diverseness among generated facial reactions (FRDiv)生成的面部反应的多样性
- Facial reaction variance (FRVar)面部反应差异
- Diversity among facial reactions generated from different speaker behaviours (FRDvs)不同说话者行为产生的面部反应的多样性
- Facial reaction realism (FRRea)面部反应真实性
- Synchrony (FRSyn)同步性
<div align="center">
<img src="https://img2023.cnblogs.com/blog/3204150/202401/3204150-20240110121337218-1965754733.png" style="zoom:80%" alt="Metric"/>
</div>


------------

## 任务实现细节
以TransVAE (baseline之一，官方参考[TEACH](https://arxiv.org/abs/2209.04066))为例
<div align="center">
<img src="https://img2023.cnblogs.com/blog/3204150/202401/3204150-20240110150345109-161683111.png" style="zoom:60%" alt="TransVAE"/>
</div>

### Offline MAFRG
1、将video、audio、emotion、3dmm随机从751帧里抽取**256帧**作为patch，包括**speaker_video_clip** [B, 256, 224, 224], **speaker_audio_clip** [B, 256, 78], **speaker_emotion** [B, 256, 25], **speaker_3dmm** [B, 256, 58], **listener_video_clip**, **listener_audio_clip**, **listener_emotion**, **listener_3dmm**, **listener_reference** (listener_video[0]，是原751帧的第一帧) [B, 1, 224, 224]；

2、将speaker_video_clip通过Conv3D得到x[B, 256, 128]，speaker_audio_clip通过Linear得到y[B, 256, 128]，将x和y拼接cat在一起再通过Linear得到**speaker_behaviour_feature**[B, 256, 128]；

3、speaker_behaviour_feature通过Linear得到x, shape不变，同时创建两个可学习参数mu_token和logvar_token， shape和x一样，并随机初始化torch.randn，然后拼接cat在一起得到x [B, 2+256, 128], 再通过一层TransformerEncoder (mask全为0，不遮蔽)，得到z [B, 2+256, 128], 令mu和logvar分别等于z[0]和z[1]，logvar再exp()和pow(0.5)得到std，对mu和std取正态分布**dist** [B, 128]，然后rsample一个值**motion_sample [B, 128]** (mu + eps*std, eps随机从标准正态分布中采样)；

4、创建time_queries（zeros[B, 256, 128]）并加上positionembedding得到tgt，并创建tgt_mask[n_head*B, 256, 256]，上三角矩阵，然后升维motion_sample至[B, 1, 128]并令其为memory，将三者送入一层TransformerDecoder得到listener_reaction，令其作为tgt和memory再加上tgt_mask送入一层TransformerDecoderTransformerDecoder得到**listener_reaction** [B, 256, 128]；

5、listener_reaction通过Linear得到**listener_3dmm_out**[B, 256, 58]，最后将listener_reaction和listener_3dmm_out拼接cat在一起得到x[B, 256, 128+58]，通过两层Linear(128+58 -> 128 -> 25)得到**listener_emotion_out** [B, 256, 25]；

6、最后输出listener_3dmm_out、listener_emotion_out、dist

### Online MAFRG
1、将video、audio、emotion、3dmm随机从751帧里抽取**256帧**作为patch（与offline task一致）；

2、设置window_size(16)，将256帧划分为256//16=16个时间段

3、第一次创建listener_reaction_3dmm（zeros[B, window_size, 224, 224]）和listener_reaction_emotion（zeros[B, window_size, 25]）；

4、当计算第i（i=1,2,..16）个时间段的listener_reaction时，先取前i个时间段（包括第i个）的speaker_video[B, (i *16), 58]和speaker_audio[B, (i *16), 78]，然后得到**speaker_behaviour_feature**[B, (i *16), 128]，同时注意**第1个时间段后续计算与offline一样，也就是说不使用下面5-8的方法**；

5、将speaker_behaviour_feature、listener_reaction_3dmm和listener_reaction_emotion拼接cat得到fuse [B, (i *16), 128+58+78]，通过Linear得到**encoded_feature** [B, (i *16), 128];

6、令past_reaction_3dmm和past_reaction_emotion为前i-1个时间段的listener_reaction_3dmm[B, (i-1 *16), 58]和listener_reaction_emotion[B, (i-1 *16), 25]；

7、将encoded_feature按offline的第3点处理方法得到**motion_sample** [B, 128]和**dist** [B, 128]，然后再将motion_sample通过offline的第4点处理方法得到**listener_reaction** [B, 16, 128]，然后将past_reaction_3dmm通过Linear得到**past_reaction_3dmm**[B, (i-1 *16), 128]并令**past_reaction_3dmm_last** = past_reaction_3dmm[:, -1] ([B, 16, 128])，也就是最接近第i个时间段的那帧；

8、创建tgt_mask [n_head*B, (i *16), (i *16)]，将listener_reaction通过Linear得到listener_reaction，shape不变，然后将listener_reaction与past_reaction_3dmm拼接cat得到**all_3dmm** [B, i *16, 128]，令all_3dmm为tgt和memory，加上tgt_mask通过一层TransformerDecoder得到listener_3dmm_out [B, i *16, 128]，然后再取第i段的16帧为新的listener_3dmm_out [B, 16, 128]，再将其送入**LSTM**(input = listener_3dmm_out, (h_0 = past_reaction_3dmm_last, c_0 = past_reaction_3dmm_last))，得到listener_3dmm_out [B, 16, 128]，最后通过Linear得到最终**listener_3dmm_out** [B, 16, 58]；
<div align="center">
<img src="https://img2023.cnblogs.com/blog/3204150/202401/3204150-20240110135315286-811971022.jpg" style="zoom:60%" alt="LSTM"/>
</div>

9、与offline第5点类似，通过listener_3dmm_out得到最终的**listener_3dmm_out** [B, 16, 58]和**listener_emotion_out** [B, 16, 25]；

10、将listener_reaction_3dmm和listener_reaction_emotion逐段cat，dist逐段append，最后得到[B, 16*16, 58/25]和[128]的数据。

------------
## Evaluate
online和offline评估过程基本一致。
- 首先会将完整的speaker_video和speaker_audio跑一次得到listener_reaction_3dmm、listener_reaction_emotion、dist并根据3dmm可视化预测listener的2d图片和3d视频
- 然后继续跑9次并将结果与第一次合并得到[B, 10, 750, 58/25]的listener_reaction_3dmm和listener_reaction_emotion，这两者参与后续的metric计算  

以下是两个子任务的官方pth可视化结果：
[Online](https://github.com/Hunan-tiger/react/blob/18db9df9cfebbae5a772893007aa91bab7ba6743/results_video/online.mp4 "Online")
[Offline](https://github.com/Hunan-tiger/react/blob/18db9df9cfebbae5a772893007aa91bab7ba6743/results_video/offline.mp4 "Offline")
