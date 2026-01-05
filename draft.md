# STDFusionNet：基于显著目标检测的红外-可见光图像融合网络

- 题目：**STDFusionNet: An Infrared and Visible Image Fusion Network Based on Salient Target Detection**  
- 期刊：**IEEE Transactions on Instrumentation and Measurement**, Vol. 70, 2021（Art. no. 5009513）  
- 作者：Jiayi Ma, Linfeng Tang, Meilong Xu, Hao Zhang, Guobao Xiao  
- DOI：10.1109/TIM.2021.3075747  
- 代码： https://github.com/jiayi-ma/STDFusionNet  

---

## 1. 摘要

- 论文提出一种**基于显著目标检测（salient target detection）的红外-可见光融合网络**，命名为 **STDFusionNet**，其目标是**保留红外图像的热目标（thermal targets）**与**可见光图像的纹理结构（texture structures）**。
- 作者引入**显著目标mask（salient target mask）**用于标注红外图像中“人或机器更关注的区域”，以此为不同信息的融合提供**空间引导（spatial guidance）**。
- 论文将显著目标mask与**特定loss函数**结合，用于指导特征的提取与重建；并指出：特征提取网络可选择性提取红外显著目标特征与可见背景纹理特征，重建网络融合并重建期望结果。
- 论文强调：显著目标mask**仅在训练阶段需要**，使得STDFusionNet在测试时是**端到端模型**；并且模型可**隐式实现显著目标检测与关键信息融合**
- 论文给出实验结论：相对“state of the arts”，其在公共数据集上可对 EN/MI/VIF/SF 指标分别取得约 **1.25% / 22.65% / 4.3% / 0.89%** 的提升，并强调算法更快。

---

## 2. 引言与动机

![Fig. 1：现有方法对有用信息的削弱示例](./../images/image-20260105095415063.png)

- 引言指出：单一传感器/单一拍摄设置得到的图像只能从有限视角描述场景，因此融合来自不同传感器/不同设置的互补图像有助于增强场景理解；其中红外-可见光融合是重要场景之一。
- 论文给出问题现象：一些现有融合方法会削弱“有用信息（useful information）”；并在示例中指出：U2Fusion 会弱化显著目标，FusionGAN 会弱化背景纹理。

![Fig. 2：STDFusionNet示例对比（GTF / DenseFuse / STDFusionNet）](./../images/image-20260105095527517.png)

- 论文在 Fig.2 的题注中说明：从左到右依次为红外、可见、传统方法GTF结果、深度方法DenseFuse结果、以及本文STDFusionNet结果；红框与绿框用于展示GTF与DenseFuse存在细节损失、边缘模糊、伪影，而STDFusionNet更好突出目标并具有丰富纹理。

---

## 3. 贡献

- 贡献1：定义融合过程中的“期望信息（desired information）”为**红外图像的显著目标**与**可见图像的背景纹理**的组合，并声称这是**首次对红外-可见光融合目标的显式定义**。
- 贡献2：将**显著目标mask**引入特定loss函数，引导网络检测红外热辐射目标并与可见背景纹理细节融合。
- 贡献3：大量实验显示优越性；并指出融合结果“看起来像高质量可见光图像且目标突出”，有助于目标识别与场景理解。

---

## 4. 方法（STDFusionNet）

![Fig. 3：STDFusionNet总体结构 + ResBlock局部结构](./../images/image-20260105095649082.png)

### 4.1 符号与“期望信息”定义

- 论文明确指出：在红外-可见光融合中，最关键的信息是**显著目标**与**纹理结构**，分别来自红外图像与可见光图像；因此将“期望信息”显式定义为：红外图像中的显著目标信息 + 可见光图像中的背景纹理结构信息。
- 论文据此提出两项关键：  
  1) 确定红外图像中的显著目标区域（通常是能发出更多热量的对象，如 pedestrians/vehicles/bunkers 所在区域）；网络需要学习从红外图像中自动检测这些区域；  
  2) 从检测到的区域准确提取期望信息并进行有效融合与重建，使融合结果在红外显著区域包含红外显著目标，在背景区域保留可见纹理。
- 在loss构建中，作者用显著目标mask $I_m$ 将“期望结果（desired result）”定义为：  
  
$$
I_d = I_m \circ I_{ir} + (1 - I_m) \circ I_{vi}
  \tag{1}
$$

  其中 $\circ$ 表示逐元素乘（elementwise multiplication）。

> 读图对应（Fig.3）：左侧的“Salient target mask”与其“背景mask（反相）”，在图中通过“逐元素乘”节点把源图像分成“显著区域”和“背景区域”。（对应式(1)中两项的分区思想）

---

### 4.2 Fig.3 总体框架：每一个模块/节点对应的实现含义

> 下面按 **原论文 Fig.3** 从左到右、从上到下逐点解释（图中包含：输入、两条特征提取分支、特征重建网络、以及训练期的mask分区loss构建）。

#### 4.2.1 输入与“分区”操作（Fig.3 左侧）

- **输入1：Visible Image（可见光图像）**，在 Fig.3 顶部作为可见分支的输入；同时在 Fig.3 下方参与与背景mask相乘以得到“可见背景区域”。（见 Fig.3 的连线与乘法节点示意）
- **输入2：Infrared Image（红外图像）**，在 Fig.3 顶部作为红外分支的输入；同时在 Fig.3 下方参与与显著mask相乘以得到“红外显著区域”。（见 Fig.3 的连线与乘法节点示意）
- **输入3：Salient target mask $I_m$**：论文说明其目的在于高亮红外图像中“能辐射大量热量”的对象（如 pedestrians/vehicles/bunkers）。
- **背景mask（Fig.3 中的反相mask）**：论文明确写到“salient target masks are inverted to obtain the background masks”。
- **逐像素乘（Fig.3 图例中的 element-wise multiplication）**：论文写到将显著mask与背景mask分别在像素级与红外/可见图像相乘，得到“source salient target regions”和“source background texture regions”。

> 对应实现：这部分就是“mask分区”，属于训练期loss构建的前处理；并不意味着mask被送进主干网络作为输入。论文强调 mask 仅用于训练期引导，不需要在测试期输入网络。

#### 4.2.2 Feature Extraction Network（Fig.3 顶部两条分支）

- 论文说明：特征提取网络采用 **pseudosiamese** 架构以“区别对待”不同模态的源图像，从而选择性地从红外图像提取显著目标特征、从可见图像提取背景纹理特征。
- Fig.3 顶部可见：可见分支与红外分支都先经过一个 **Conv 5×5**（图中标注“Conv 5×5”）和一个 **lrelu**（leaky rectified linear unit），然后接 **ResBlock×3**（图中标注 ResBlock、重复3次）。
- 论文解释：该 pseudosiamese 架构中两条特征提取网络“具有相同架构”，但“参数独立训练”，原因是红外与可见光图像属性不同。

> 读图对应（Fig.3）：两条分支的结构相同（Common layer + ResBlocks），但论文强调它们不是共享权重的严格Siamese，而是“pseudosiamese”（架构同、参数独立）。

#### 4.2.3 Feature Reconstruction Network（Fig.3 顶部右侧）

- Fig.3 中在两条特征提取分支之后进入 **Feature Reconstruction Network**（虚线框），其内部由 **ResBlock×4** 组成，并最终输出融合图像 $I_f$（图中标注 “Fused Image”）。
- 论文写明：特征重建网络的输入是“红外卷积特征与可见卷积特征在通道维度上的拼接（concatenation in the channel dimension）”。
- 论文补充：重建网络最后一层使用 **Tanh** 激活，以保证输出图像取值范围与输入源图像一致。

#### 4.2.4 Loss Function（Fig.3 中部“Loss Function”方块）

- Fig.3 的 Loss Function 方块中用简写表达了 “$L = L_p + L_{grad}$” 的思想；正文进一步说明其loss由两类损失构成：**pixel loss**（约束融合图像像素强度一致性）与 **gradient loss**（促使融合图像包含更多细节信息）。
- 论文强调：pixel/gradient loss 都分别在**显著区域**与**背景区域**构建，并结合显著mask $I_m$ 把融合图像划分为 $I_m\circ I_f$（显著区域）与 $(1-I_m)\circ I_f$（背景区域）。

---

### 4.3 Loss 公式细读（式(2)-(6)）：像素一致 + 梯度一致 + 显著/背景分区

#### 4.3.1 像素损失 Pixel loss（显著区域 / 背景区域）

- 显著区域像素损失：  
  
$$
L^{pixel}_{salient}=\frac{1}{HW}\left\| (I_m\circ I_f)-(I_m\circ I_{ir})\right\|_1
  \tag{2}
$$

- 背景区域像素损失：  
  
$$
L^{pixel}_{back}=\frac{1}{HW}\left\| ((1-I_m)\circ I_f)-((1-I_m)\circ I_{vi})\right\|_1
  \tag{3}
$$

- 论文说明：$\|\cdot\|_1$ 为 L1 范数，$H,W$ 分别为图像高和宽。

#### 4.3.2 梯度损失 Gradient loss（显著区域 / 背景区域）

- 论文写明：梯度算子 $\nabla$ 使用 **Sobel operator** 来计算图像梯度。
- 显著区域梯度损失：  
  
$$
L^{grad}_{salient}=\frac{1}{HW}\left\| (I_m\circ \nabla I_f)-(I_m\circ \nabla I_{ir})\right\|_1
  \tag{4}
$$

- 背景区域梯度损失：  
  
$$
L^{grad}_{back}=\frac{1}{HW}\left\| ((1-I_m)\circ \nabla I_f)-((1-I_m)\circ \nabla I_{vi})\right\|_1
  \tag{5}
$$


#### 4.3.3 总损失（式(6)）：区域权重 + 同区域内 pixel/grad 等权

- 论文指出：与以往方法不同，作者在“同一个区域内”对 pixel loss 与 gradient loss **同等对待（equally）**，因此最终loss为：  
  
$$
L = (L^{pixel}_{back}+L^{grad}_{back})+\alpha(L^{pixel}_{salient}+L^{grad}_{salient})
  \tag{6}
$$

- 论文解释：$\alpha$ 是控制背景区域与显著区域 loss 平衡的权重；并指出通过在 loss 中引入显著区域约束，模型具有“自动检测并提取红外显著目标”的能力。

---

### 4.4 显著目标mask的获取与在训练中如何用（对应 Fig.3 下半部分 + Fig.4）

 ![Fig. 4：源图像与mask示例（TNO）](./../images/image-20260105100041764.png)

- 论文说明：使用 **LabelMe toolbox** 标注红外图像中的显著目标并转成二值mask；之后将mask取反得到背景mask。
- 随后：  
  - 将显著mask与背景mask分别在像素级与红外/可见图像相乘，得到“源显著区域”和“源背景纹理区域”；  
  - 将融合图像同样与显著mask/背景mask相乘，得到“融合显著区域”和“融合背景区域”；  
  - 最终用这些区域去构造特定loss，从而引导网络隐式实现显著目标检测与信息融合。
- 论文强调：显著目标mask**仅用于训练引导**，测试阶段不需要输入网络，因此模型端到端。

---

### 4.5 网络结构细节（Fig.3 顶部/底部 ResBlock 逐点说明）

#### 4.5.1 Feature Extraction Network（两条分支，pseudosiamese）

- 论文指出：特征提取部分包含两条网络（红外/可见），二者**架构相同但参数独立训练**，以适应不同模态图像的属性差异。
- 每条特征提取网络由：  
  - **Common layer**：一个 5×5 卷积层 + 一个 leaky ReLU 激活层；  
  - **3 个 ResBlocks**：用于增强提取的信息（论文写“reinforce the extracted information”）。

#### 4.5.2 ResBlock（Fig.3 底部局部结构：每一个点/算子）

> Fig.3 下方给出了 ResBlock 的结构示意，论文正文也逐点描述：

- ResBlock 有两条路径：  
  **主分支（上路）**：Conv1(1×1) → lrelu → Conv2(3×3) → lrelu → Conv3(1×1)；  
  **旁路（下路）**：identity conv(1×1)。  
  两路输出在“+”节点处相加后，再过一个 lrelu 输出。
- 论文写明：除 Conv2 为 3×3 外，其余卷积核大小均为 1×1；Conv1/Conv2 后接 leaky ReLU；Conv3 与 identity conv 输出先相加再接 leaky ReLU。
- 论文解释：identity conv 的设计用于解决 ResBlock 输入与输出维度不一致的问题。

#### 4.5.3 Feature Reconstruction Network（融合与重建）

- 论文写明：特征重建网络由 **4 个 ResBlocks** 组成；其输入是红外与可见分支特征的通道拼接；最后一层使用 Tanh 激活以保证输出范围与输入一致。

#### 4.5.4 padding/stride（与“无下采样”相关）

- 论文强调：信息丢失对融合任务是灾难性的，因此 STDFusionNet 的所有卷积层采用 **padding = SAME** 与 **stride = 1**；由此网络不引入下采样，融合图像尺寸与源图像一致。

---

## 5. 方法实现：按 Fig.3 + 式(1)-(6) 整理的训练/推理流程（代码占位）

> 这一部分把 Fig.3 中“上半部分网络前向”与“下半部分mask分区loss”拼成可实现的流程；占位代码仅对应论文明确描述的组件。

**数据预处理（归一化到 [-1,1]、裁剪 stride=24、patch=128×128；测试不裁剪）**  
来自 `utils.input_setup` 和 `train.py`。训练时每张源图/掩码都按 stride=24 滑窗裁成 128×128 patch，并用 `(imread(...) - 127.5)/127.5` 归一化到 [-1,1]。  

```python
def input_setup(sess, config, data_dir, index=0):
    if config.is_train:
        data = prepare_data(sess, dataset=data_dir)
    else:
        data = prepare_data(sess, dataset=data_dir)

    sub_input_sequence = []

    if config.is_train:
        for i in range(len(data)):
            input_ = (imread(data[i]) - 127.5) / 127.5
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                    sub_input_sequence.append(sub_input)
    ...
```
- 对齐注释：`config.image_size=128`、`config.stride=24`（见 5.5 训练脚本）；对应论文训练细节“crop 128×128 with stride 24 得到 6921 patch”与“输入归一化到 [-1,1]”，以及 Fig.3 左侧输入节点尺寸保持不变（无下采样）。
- 可见/红外/掩码都通过 `input_setup(..., "Train_vi") / ("Train_ir") / ("Train_ir_mask_blur")` 进入 `train()`（见 5.5），与 Fig.3 下半部分“mask 仅用于 loss 构造”保持一致。

### 5.1 前向传播（Inference / Training都需要）

```python
class STDFusionNet:
    def vi_feature_extraction_network(self, vi_image):
        with tf.compat.v1.variable_scope('vi_extraction_network'):
            with tf.compat.v1.variable_scope('conv1'):
                weights = tf.compat.v1.get_variable("w", [5, 5, 1, 16],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.compat.v1.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(vi_image, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                conv1 = tf.nn.leaky_relu(conv1)
            block1_input = conv1
            ...  # ResBlock ×3（见下节）
        return encoding_feature

    def ir_feature_extraction_network(self, ir_image):
        with tf.compat.v1.variable_scope('ir_extraction_network'):
            ...  # 与上完全同构，但参数独立
        return encoding_feature

    def feature_reconstruction_network(self, feature):
        with tf.compat.v1.variable_scope('reconstruction_network'):
            block1_input = feature
            ...  # ResBlock ×4
            block4_output = tf.nn.tanh(conv3 + identity_conv)
            fusion_image = block4_output
        return fusion_image

    def STDFusion_model(self, vi_image, ir_image):
        with tf.variable_scope("STDFusion_model"):
            vi_feature = self.vi_feature_extraction_network(vi_image)
            ir_feature = self.ir_feature_extraction_network(ir_image)
            feature = tf.concat([vi_feature, ir_feature], axis=-1)
            f_image = self.feature_reconstruction_network(feature)
        return f_image
```
- 代码来源：`train_network.py`。
- 对齐注释：
  - `vi_feature_extraction_network` / `ir_feature_extraction_network`：对应 Fig.3 顶部两条 **pseudosiamese** 分支（同构不同参数）。`conv1` 为 Fig.3 中的 “Conv 5×5 + lrelu”（common layer），后续 `block1/2/3` 为 “ResBlock×3”。
  - `feature_reconstruction_network`：对应 Fig.3 右上虚线框 “Feature Reconstruction Network” 的 4 个 ResBlock，`tf.concat([...], axis=-1)` 对应 Fig.3 中红外/可见特征的 **channel concat** 箭头。
  - `tf.nn.tanh` 终层与论文“最后一层使用 Tanh 以保证输出范围与输入一致”逐点对齐；所有卷积 `padding='SAME'`, `strides=[1,1,1,1]` 对应 “无下采样、保持尺寸”。
  - `STDFusion_model`：前向 `If = R( concat( F_ir(Iir), F_vi(Ivi) ) )`，对应 Fig.3 顶部整体流程。

### 5.2 ResBlock 占位（严格对应 Fig.3 局部结构）

```python
with tf.compat.v1.variable_scope('block2'):
    with tf.compat.v1.variable_scope('conv1'):
        weights = tf.compat.v1.get_variable("w", [1, 1, 16, 16],
                                            initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.compat.v1.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
        conv1 = tf.nn.leaky_relu(conv1)

    with tf.compat.v1.variable_scope('conv2'):
        weights = tf.compat.v1.get_variable("w", [3, 3, 16, 16],
                                            initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.compat.v1.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
        conv2 = tf.nn.leaky_relu(conv2)
    with tf.compat.v1.variable_scope('conv3'):
        weights = tf.compat.v1.get_variable("w", [1, 1, 16, 32],
                                            initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.compat.v1.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
    with tf.variable_scope('identity_conv'):
        weights = tf.compat.v1.get_variable("w", [1, 1, 16, 32],
                                            initializer=tf.truncated_normal_initializer(stddev=1e-3))
        identity_conv = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME')
    block2_output = tf.nn.leaky_relu(conv3 + identity_conv)
```
- 代码来源：`train_network.py`（可见分支 block2 示意，红外与重建分支结构同型）。
- 对齐注释：
  - “主分支”Conv1/Conv2/Conv3 核大小依次 1×1、3×3、1×1，Conv1/Conv2 后接 leaky ReLU；“旁路” identity conv 为 1×1，用于通道升维（16→32）。
  - `conv3 + identity_conv` 后整体再过 `leaky_relu`，对应 Fig.3 ResBlock 底部 “+” 后接 lrelu。
  - kernel padding=‘SAME’、stride=1 保持尺寸，对应论文“无下采样”。

### 5.3 Sobel 梯度算子占位（对应式(4)(5)中的 ∇，论文写用 Sobel operator）

```python
def gradient(input):
    filter1 = tf.reshape(tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]), [3, 3, 1, 1])
    filter2 = tf.reshape(tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]), [3, 3, 1, 1])
    Gradient1 = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='SAME')
    Gradient2 = tf.nn.conv2d(input, filter2, strides=[1, 1, 1, 1], padding='SAME')
    Gradient = tf.abs(Gradient1) + tf.abs(Gradient2)
    return Gradient
```
- 代码来源：`utils.py`。
- 对齐注释：`filter1/filter2` 正是 Sobel 水平/垂直核，对应论文“we employ the Sobel operator to compute the gradient”；`Gradient` 即公式中的 $\nabla I$，用于 Eq.(4)(5) 的区域梯度损失。

### 5.4 Loss 计算占位（对应式(2)-(6)，含 mask 分区）

```python
with tf.name_scope('g_loss'):
    self.ir_mask = (self.ir_mask + 1) / 2.0
    self.ir_p_loss_train = tf.multiply(self.ir_mask, tf.abs(self.fusion_images - self.ir_images))
    self.vi_p_loss_train = tf.multiply(1 - self.ir_mask, tf.abs(self.fusion_images - self.vi_images))
    self.ir_grad_loss_train = tf.multiply(self.ir_mask, tf.abs(gradient(self.fusion_images) - gradient(self.ir_images)))
    self.vi_grad_loss_train = tf.multiply(1 - self.ir_mask, tf.abs(gradient(self.fusion_images) - gradient(self.vi_images)))

    self.ir_p_loss = tf.reduce_mean(self.ir_p_loss_train)
    self.vi_p_loss = tf.reduce_mean(self.vi_p_loss_train)
    self.ir_grad_loss = tf.reduce_mean(self.ir_grad_loss_train)
    self.vi_grad_loss = tf.reduce_mean(self.vi_grad_loss_train)
    self.g_loss_2 = 1 * self.vi_p_loss + 1 * self.vi_grad_loss + 7 * self.ir_p_loss + 7 * self.ir_grad_loss
```
- 代码来源：`model.py` 的 `build_model`。
- 对齐注释：
  - `self.ir_mask = (mask+1)/2` 将训练集存储的 [-1,1] 掩码还原为二值 $\{0,1\}$（论文 mask 二值化），`1 - self.ir_mask` 对应论文“inverted to obtain the background masks”。
  - `tf.multiply(self.ir_mask, |If - Iir|)`、`tf.multiply(1 - self.ir_mask, |If - Ivi|)`：对应 Eq.(2)(3) 的 $I_m \circ I_f - I_m \circ I_{ir}$ 与 $(1-I_m)\circ I_f - (1-I_m)\circ I_{vi}$，L1 范数由 `tf.abs` + `tf.reduce_mean`（即 $\frac{1}{HW}\|\cdot\|_1$）实现。
  - `gradient(...)` 调用 Sobel（见 5.3），对应 Eq.(4)(5) 中 $\nabla I$；同样用显著/背景 mask 做逐元素乘，符合 Fig.3 下半部分“Element-wise Multiplication”节点。
  - `self.g_loss_2 = vi_pixel + vi_grad + 7*ir_pixel + 7*ir_grad`：对应 Eq.(6) 中背景区域（对应可见图）系数 1，显著区域系数 $\alpha=7$；pixel/grad 在同一区域等权相加。

### 5.5 训练流程（对应论文“Training Details”）

```python
flags.DEFINE_integer("epoch", 30, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 128, "The size of image to use [33]")
flags.DEFINE_integer("stride", 24, "The size of stride to apply input image [14]")
flags.DEFINE_float("learning_rate", 1e-3, "The learning rate of gradient descent algorithm [1e-4]")
...
with tf.name_scope('train_step'):
    self.train_generator_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total, var_list=self.g_vars)
...
for ep in range(config.epoch):
    lr = self.init_lr if ep < self.decay_epoch else self.init_lr * (config.epoch - ep) / (config.epoch - self.decay_epoch)
    batch_idxs = len(train_data_ir) // config.batch_size
    for idx in range(0, batch_idxs):
        batch_vi_images = train_data_vi[idx * config.batch_size: (idx + 1) * config.batch_size]
        batch_ir_images = train_data_ir[idx * config.batch_size: (idx + 1) * config.batch_size]
        batch_ir_mask = train_data_ir_mask[idx * config.batch_size: (idx + 1) * config.batch_size]
        batch_ir_mask = (batch_ir_mask + 1.0) / 2.0
        _, err_g, ... = self.sess.run(
            [self.train_generator_op, self.g_loss_total, ...],
            feed_dict={self.vi_images: batch_vi_images, self.ir_images: batch_ir_images,
                       self.ir_mask: batch_ir_mask, self.lr: lr})
```
- 代码来源：`train.py`（超参定义）与 `model.py`（`STDFusion.train` 循环）。
- 对齐注释：
  - `epoch=30`、`batch_size=32`、`learning_rate=1e-3`、`stride=24`、`image_size=128` 与论文训练细节完全一致；`AdamOptimizer` 对应论文“optimizer is Adam in TensorFlow”。
  - 训练数据由 20 对 TNO 图像（目录 `Train_ir`/`Train_vi` 各 20 张）经 5.1 的裁剪产生 patch（论文统计 6921 对）；测试分支未裁剪，符合“测试不裁剪”。
  - `batch_ir_mask = (mask+1)/2` 再喂入 loss，与 Fig.3 中 mask 仅用于 loss（不进入网络）一致。

---

## 6. 实验设置（Datasets / Metrics / Training Details）

### 6.1 数据集（TNO & RoadScene）

- 论文在实验中使用两个数据集：TNO 与 RoadScene。
- TNO：包含 60 对红外/可见图像，分为三个序列，分别含 19、23、32 对；Fig.4 给出典型源图像与对应mask示例。
- RoadScene：由 Xu 等基于 FLIR 视频发布，包含 221 对对齐的红外/可见图像，场景包含道路、车辆、行人，并被描述为缓解“样本少与低分辨率”的挑战。

### 6.2 指标（EN / MI / VIF / SF）

- 论文选择四个常用指标：EN、MI、VIF、SF，并在文中给出其定义公式；并说明客观评价是对主观评价的补充。
- 论文给出 SF 的定义并指出：SF 大意味着融合图像含有更丰富的纹理与细节，从而性能更好。

### 6.3 训练细节（Training Details）

- 训练：在 TNO 上训练，训练图像对数量为 20；为获得更多数据，设置 stride=24 进行裁剪，每个 patch 大小 128×128，得到 6921 对 patch。
- 测试：在 TNO 选 20 对做对比实验，在 RoadScene 选 20 对做泛化实验；并强调测试时源图像直接输入网络、不做裁剪。
- 归一化与优化：源图像归一化到 [-1,1]；使用 Adam；实现平台 TensorFlow；batch size=32，iteration=30，学习率 1e-3。
- $\alpha$ 取值：论文观察到显著区域只占红外图像很小比例，因此为平衡显著/背景区域的loss，设 $\alpha=7$。硬件：NVIDIA TITAN V GPU + 2.00-GHz Intel Xeon Gold 5117 CPU。

---

## 7. 对比实验结果（论文原文描述 + 图表编号定位）

### 7.1 对比方法（9个）

- 论文比较9种方法：传统方法 GTF、MDLatLRR；深度方法 DenseFuse、NestFuse、FusionGAN、GANMcC、IFCNN、PMGI、U2Fusion，并说明这些方法实现公开且参数按原文设置。

---

### 7.2 TNO：主观结果（Figs.5–8）与论文给出的观察

![image-20260105100704044](./../images/image-20260105100704044.png)
![image-20260105100742742](./../images/image-20260105100742742.png)
![image-20260105100756196](./../images/image-20260105100756196.png)
![image-20260105100819892](./../images/image-20260105100819892.png)

- 论文在 TNO 上选择四个典型图像对（bench、Kaptein_1123、Kaptein_1654、Tree_4915）做主观评价，并在 Fig.5 中用红框标注显著区域进行放大对比。
- 论文描述（Fig.5）：MDLatLRR 会丢失热辐射目标信息；DenseFuse/IFCNN/U2Fusion 虽保留热辐射目标信息，但受到严重噪声污染（来源为可见图像信息）。
- 论文总结四个场景：STDFusionNet 不仅能有效突出显著目标，还在保持背景纹理细节方面有明显优势；并举例说明：Kaptein_1123 中树枝纹理最清晰且天空不被热辐射污染；Kaptein_1654 中背景路灯与可见图几乎一致；Tree_4915 中其他方法几乎无法区分灌木与背景，而 STDFusionNet 能突出红外目标并区分灌木。
- 论文指出：这种“选择性保留红外显著目标 + 可见纹理细节”的表现，主要得益于训练时人工提取的显著目标mask与构造的loss函数。

---

### 7.3 TNO：客观结果（Fig.9 + Table I）与论文对指标的解释

![image-20260105100846647](./../images/image-20260105100846647.png)

![image-20260105100909822](./../images/image-20260105100909822.png)

- Fig.9 的题注说明：在 TNO 的 20 对图像上对 EN/MI/VIF/SF 做曲线对比；曲线上一点 (x,y) 表示有 (100*x)% 的图像对的指标值不超过 y；并列出用于比较的9种方法名称。
- 论文对 TNO 的定量结论：在四个指标中，STDFusionNet 在 EN、MI、VIF 三项上优势显著；SF 指标仅以很小差距落后于 IFCNN。
- 论文强调：STDFusionNet 在 VIF 上几乎所有图像对都取最高值，这与主观评价一致，表明其融合图像具有更好的视觉效果；并解释 EN 最大说明信息更丰富、MI 最大说明从源图像传递的信息更多；SF 虽非最佳但“可比结果”表明融合结果具备足够梯度信息。

---

### 7.4 泛化实验（RoadScene）：彩色可见图像的融合策略 + 论文观察

![image-20260105100956795](./../images/image-20260105100956795.png)
![image-20260105101015300](./../images/image-20260105101015300.png)
![image-20260105101038599](./../images/image-20260105101038599.png)
![image-20260105101052304](./../images/image-20260105101052304.png)

- 泛化设置：使用 RoadScene 测试在 TNO 上训练的模型，以评估泛化能力。
- 因 RoadScene 可见图像为彩色，论文采用特定融合策略以保色：RGB→YCbCr；将 Y 通道与灰度红外图像进行融合；再用可见图的 Cb/Cr 做逆变换恢复 RGB 融合结果。
- 论文对 Figs.10–13 的观察：STDFusionNet 能选择性保留红外与可见的有用信息；其融合图像在显著区域非常接近红外图像，且背景区域几乎完整保留可见纹理结构；而其他方法虽然能突出目标，但融合背景“极不令人满意”，例如天空被热信息严重污染，影响对时间/天气判断；同时其他方法对墙面文字、车辆、树桩、栅栏、路灯等背景细节保留不佳，STDFusionNet 则能有效保留背景细节并维持/增强目标对比度。

![image-20260105101108440](./../images/image-20260105101108440.png)

- 论文对 RoadScene 的定量结论：STDFusionNet 在 MI/VIF/SF 的平均值最好；EN 指标仅以很小差距落后 NestFuse；并据此认为其具有良好泛化性，受成像传感器特性影响较小。

---

### 7.5 效率对比（Table II）与论文结论

![image-20260105101125166](./../images/image-20260105101125166.png)

- 论文指出：运行效率也是重要因素；Table II 给出在 TNO 与 RoadScene 上不同方法的平均运行时间；并指出深度方法因 GPU 加速在运行时间上有优势，尤其是 STDFusionNet；传统方法耗时更长，MDLatLRR 因分解过程尤其耗时。
- 论文结论：STDFusionNet 在两数据集上具有**最小平均运行时间**与**最小标准差**，说明网络对不同分辨率源图像具有鲁棒性并证明了结构设计的效率。

---

## 8. “显著目标检测”可视化（Fig.15）

![image-20260105101153245](./../images/image-20260105101153245.png)

- 论文写到：STDFusionNet 可“隐式”实现显著目标检测，并给出可视化：展示红外图像的显著区域，以及“从融合结果中减去可见背景区域”的差分结果。
- 论文指出：差分结果与红外显著区域基本一致，且存在“额外的热显著目标”被方法检测到的现象，从而表明 STDFusionNet 能隐式执行显著目标检测。

---

## 9. 消融实验（Fig.16 + Table III）：期望信息定义 & 梯度loss

![image-20260105101207210](./../images/image-20260105101207210.png)

![image-20260105101222761](./../images/image-20260105101222761.png)

### 9.1 w/o desired information（去掉“期望信息定义”的消融）

- 论文说明：为验证“期望信息定义”的合理性，在 TNO 上训练两种模型，主要差异是是否将显著目标mask引入loss；当移除显著mask后，不需要区分显著/背景区域，因此将 $\alpha$ 设为 1。
- 论文在 Fig.16 中描述：有期望信息定义时，STDFusionNet 的结果能突出显著目标并维持背景纹理；不使用期望信息定义时，网络以“coarse manner”进行融合，导致显著区域的热辐射信息与背景纹理信息都不能很好保留。

### 9.2 w/o gradient loss（去掉梯度loss的消融）

- 论文在 Fig.16 附近写到：移除 gradient loss 时，显著区域几乎没有纹理信息，显著目标形状出现严重扭曲，背景区域也出现伪影；并且在 Table III 中除 SF 外其他指标呈下降趋势，论文据此强调 gradient loss 的重要性：它能确保融合图像中显著目标的纹理清晰度（texture sharpness）。

---

## 10. 结论（Conclusion）

- 论文总结：提出 STDFusionNet，并将红外-可见光融合的期望信息显式定义为“红外显著区域 + 可见背景区域”；在此基础上把显著目标mask引入loss以精确引导网络优化。
- 论文指出：模型可隐式完成显著目标检测与信息融合，融合结果既包含显著热目标也具有丰富背景纹理；大量主观与客观实验验证其优越性，并且运行速度更快。

---
