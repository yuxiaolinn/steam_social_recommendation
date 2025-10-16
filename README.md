# steam_social_recommendation

# Steam游戏推荐GNN系统

基于图神经网络的Steam游戏推荐系统，实现了 **98.70% AUC** 和 **99.74% PR-AUC** 的优异性能。


## 📖 项目简介

本项目实现了一个基于图神经网络（GNN）的Steam游戏推荐系统。系统利用steam社区评论数据、情感分析和 NLP 主题特征抓取，提供精准的游戏推荐服务。

### 核心特性

- **高性能表现**：AUC达到 **0.9870**，PR-AUC达到 **0.9974**
- **增强边权重**：多因素边权重计算，考虑相对游戏时长、情感分析和评论质量
- **丰富节点特征**：26维用户特征和14维游戏特征
- **可扩展架构**：支持20,001个用户、734个游戏和23,077条交互
- **GPU加速**：针对CUDA环境优化

## 🎯 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      输入层                                  │
│  用户特征 (4维) + 游戏特征 (4维)                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    嵌入层                                    │
│  用户嵌入 (4D→64D) + 游戏嵌入 (4D→64D)                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  图卷积层                                    │
│  GCN第1层 (64D→64D) + GCN第2层 (64D→64D)                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   推荐头                                     │
│  多层感知机 (128D→64D→32D→1D) + Sigmoid激活                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
                用户-游戏交互概率 (0-1)
```

**模型参数量**：19,329个

## 📊 数据集

### 输入数据

| 数据集 | 描述 | 规模 |
|--------|------|------|
| `game_features.csv` | 从正面评论中提取的游戏主题特征 | 571个游戏 |
| `steam_user_theme_features.csv` | 用户主题偏好特征 | 20,001个用户 |
| `edge_weights.csv` | 用户-游戏交互权重 | 23,077条边 |
| `active_comments_subset.csv` | 训练数据（近期活跃用户） | 70,982条评论 |
| `english_comments_subset.csv` | 测试数据（英文评论） | 23,077条评论 |

### 数据统计

- **图密度**：0.1572%（推荐系统的典型水平）
- **用户分布**：
  - 90.9%的用户只玩1个游戏
  - 6.7%的用户玩2个游戏
  - 2.4%的用户玩3个以上游戏
  - 最多：单个用户玩150个游戏
- **游戏热度**：
  - 平均：每个游戏31.44个用户
  - 中位数：每个游戏2个用户
  - 最多：最热门游戏有3,874个用户

## 🔧 特征增强

### 1. 增强边权重

边权重计算采用混合方法，结合三个因素：

```python
核心权重 = 相对游戏时长归一化 × 情感归一化
边权重 = 0.8 × 核心权重 + 0.2 × 质量归一化
```

**组成部分**：

#### 相对游戏时长权重（通过核心权重占60%）
基于用户游戏时长与该游戏平均时长的比值：
- `> 2.0倍平均值` → 1.0（非常喜欢）
- `> 1.5倍平均值` → 0.9（很喜欢）
- `> 1.0倍平均值` → 0.7（比较喜欢）
- `> 0.5倍平均值` → 0.5（一般喜欢）
- `< 0.5倍平均值` → 0.2（不太喜欢）

#### 情感分析权重（通过核心权重占30%）
基于用户评论的VADER情感分析：
- `compound > 0.05` → 1.0（正面）
- `-0.05 ≤ compound ≤ 0.05` → 0.5（中性）
- `compound < -0.05` → 0.0（负面）

#### 评论质量权重（20%）
- `review_quality`：评论长度归一化
- `helpfulness`：有用投票比例
- `quality_norm = 0.5 × review_quality + 0.5 × helpfulness`

### 2. 用户特征（26维）

**主题偏好**（4维）：
- `music_sound_normalized`：音乐/音效偏好
- `story_narrative_normalized`：故事/叙事偏好
- `gameplay_mechanics_normalized`：游戏机制偏好
- `visuals_graphics_normalized`：视觉/画面偏好

**行为特征**：
- `num_reviews`：评论数量
- `total_playtime`：总游戏时长
- `recent_playtime`：最近两周游戏时长
- `avg_playtime_per_game`：平均每游戏时长
- `review_helpfulness`：评论平均有用性
- `avg_review_length`：平均评论长度

### 3. 游戏特征（14维）

**主题特征**（4维）：
- `music_sound`、`story_narrative`、`gameplay_mechanics`、`visuals_graphics`

**聚合特征**：
- `avg_rating`：平均评分（正面评论比例）
- `review_count`：评论数量
- `avg_playtime`：所有用户的平均游戏时长
- `helpfulness_score`：评论平均有用性
- `unique_users`：唯一用户数
- `top_topic_keywords`：评论中的关键主题词

## 🚀 快速开始

### 环境要求

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn tqdm
```

### 安装

```bash
git clone https://github.com/yourusername/steam-gnn-recommendation.git
cd steam-gnn-recommendation
```

### 训练流程

#### 步骤1：增强特征和边权重

```python
# 运行特征增强脚本
python adjust_edge_weights_features.py
```

生成文件：
- `enhanced_edge_weights.csv`：使用混合公式计算的增强边权重
- `enhanced_user_features.csv`：包含行为数据的增强用户特征
- `enhanced_game_features.csv`：包含聚合统计的增强游戏特征

#### 步骤2：训练GNN模型

```python
# 训练GNN模型
python train_gnn_vectorized.py
```

程序将：
1. 加载增强数据
2. 创建二部图结构
3. 训练GNN模型（100轮）
4. 在测试集上评估
5. 生成可视化图表
6. 保存训练好的模型到 `steam_gnn_model.pth`

### 生成推荐

```python
import torch
from train_gnn_vectorized import SteamGNN

# 加载训练好的模型
model = SteamGNN(4, 4, hidden_dim=64, num_layers=2)
model.load_state_dict(torch.load('steam_gnn_model.pth'))
model.eval()

# 获取嵌入向量
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, data.edge_weight, num_users)

# 为用户推荐游戏
def recommend_games(user_id, top_k=10):
    user_idx = user_to_idx.get(user_id)
    if user_idx is None:
        return []
    
    user_emb = embeddings[user_idx]
    scores = []
    
    for game_id, game_idx in game_to_idx.items():
        game_emb = embeddings[game_idx]
        score = model.predict_interaction(
            user_emb.unsqueeze(0), 
            game_emb.unsqueeze(0)
        )
        scores.append((game_id, score.item()))
    
    # 按分数排序并返回top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# 使用示例
recommendations = recommend_games(76561197985437504, top_k=10)
for game_id, score in recommendations:
    print(f"游戏 {game_id}: 推荐分数 {score:.4f}")
```

## 📈 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **AUC** | **0.9870** | ROC曲线下面积 - 分类性能 |
| **PR-AUC** | **0.9974** | 精确率-召回率AUC - 不平衡数据表现 |
| **参数量** | 19,329 | 模型总参数数 |
| **训练时间** | ~5分钟 | Kaggle GPU (Tesla P100) |
| **用户数** | 20,001 | 用户节点数 |
| **游戏数** | 734 | 游戏节点数 |
| **边数** | 23,077 | 用户-游戏交互数 |

## 🛠️ 技术细节

### 模型架构

- **基础模型**：图卷积网络（GCN）
- **隐藏维度**：64
- **网络层数**：2层GCN
- **激活函数**：ReLU
- **Dropout**：0.3
- **损失函数**：二元交叉熵
- **优化器**：Adam（学习率0.01）
- **训练策略**：正负样本采样

### 训练细节

- **正样本**：23,077个实际用户-游戏交互
- **负样本**：23,077个随机生成的非交互样本
- **训练轮数**：100轮
- **批处理**：全批次训练
- **设备**：CUDA GPU

## 📁 项目结构

```
steam-gnn-recommendation/
├── README.md                              # 英文文档
├── README_CN.md                           # 中文文档
├── data/                                  # 数据目录
│   ├── game_features.csv
│   ├── steam_user_theme_features.csv
│   ├── edge_weights.csv
│   ├── active_comments_subset.csv
│   └── english_comments_subset.csv
├── src/                                   # 源代码目录
│   ├── adjust_edge_weights_features.py    # 步骤1：特征增强
│   ├── train_gnn_vectorized.py            # 步骤2：GNN训练
│   └── visualize_results.py               # 可视化工具
├── utils/                                 # 工具脚本
│   ├── diagnose_data_issue.py             # 数据诊断
│   ├── analyze_edge_structure.py          # 边结构分析
│   └── analyze_edge_user_relationship.py  # 用户-游戏关系分析
└── models/                                # 模型目录
    └── steam_gnn_model.pth                # 训练好的模型
```

## 🔬 方法论

### 1. 数据采集与预处理
- 使用微调的RoBERTa模型从正面评论中提取游戏特征
- 使用VADER进行情感分析
- 从评论历史中聚合用户主题偏好

### 2. 特征工程
- **用户特征**：主题偏好 + 行为模式
- **游戏特征**：主题特性 + 聚合统计
- **边权重**：结合游戏时长、情感和质量的混合计算

### 3. 图构建
- **节点**：用户（20,001）+ 游戏（734）
- **边**：加权的用户-游戏交互（23,077）
- **图类型**：异构二部图

### 4. 模型训练
- 基于GCN的架构，包含2个卷积层
- 使用正负样本采样进行二分类
- Adam优化器，学习率0.01
- 训练100轮，支持早停

### 5. 模型评估
- 在英文评论子集上测试
- 评估指标：AUC、PR-AUC
- 负采样实现平衡评估

## 📊 结果可视化

系统生成全面的可视化图表，包括：

1. **模型性能**
   - AUC和PR-AUC柱状图
   - 训练损失曲线
   - 预测分布直方图

2. **边权重分析**
   - 原始vs增强权重对比
   - 权重分布统计
   - 权重改进分析

3. **特征分析**
   - 用户主题偏好分布
   - 游戏特征分布
   - 特征相关性热力图

4. **图结构**
   - 用户-游戏网络可视化
   - 节点度分布
   - 边权重与流行度关系

## 🎓 核心创新

### 1. 相对游戏时长归一化
使用相对游戏时长而非绝对时长，考虑不同游戏类型的差异（例如，短小的解谜游戏 vs 长篇RPG）。

### 2. 混合边权重公式
结合乘法核心权重（游戏时长 × 情感）和加法质量奖励，平衡严格要求与质量激励。

### 3. 向量化数据处理
避免iterrows()的陷阱，使用pandas merge操作，确保数据类型一致性并提高处理速度。

### 4. 多维特征融合
整合基于NLP分析的主题特征和基于用户活动的行为特征。

## 🐛 故障排除

### 常见问题

**问题**：大整数KeyError
- **解决方案**：使用向量化操作代替iterrows()
- **文件**：使用 `train_gnn_vectorized.py` 而非循环版本

**问题**：ID数据类型不匹配
- **解决方案**：在读取CSV时指定正确的dtype或使用字符串→int64转换
- **代码**：`pd.read_csv(file, dtype={'author_steamid': 'int64', 'appid': 'int64'})`

**问题**：训练时内存不足
- **解决方案**：使用GPU、减小批次大小或采样边
- **提示**：使用 `nvidia-smi` 监控GPU内存

## 📚 依赖库

```python
# 核心库
torch>=2.0.0
torch-geometric>=2.3.0
pandas>=1.5.0
numpy>=1.23.0

# 预处理
scikit-learn>=1.2.0
vaderSentiment>=3.3.2
transformers>=4.30.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.1

# 工具
tqdm>=4.65.0
```

## 🔬 数据来源

数据集基于Steam用户评论，具有以下特点：

- **评论时期**：近2周活跃用户
- **语言**：仅英文评论
- **情感分析**：VADER情感分数
- **主题分析**：基于RoBERTa的多标签分类
- **主题类别**：音乐/音效、故事/叙事、游戏机制、视觉/画面

## 📈 性能详解

### 训练性能
- **最终训练损失**：~0.12（已收敛）
- **训练时间**：在Tesla P100上约5分钟
- **GPU显存占用**：~2GB

### 评估性能
- **AUC分数**：0.9870
  - 优秀的二分类性能
  - 高度的正负交互区分能力
- **PR-AUC分数**：0.9974
  - 出色的精确率-召回率平衡
  - 对类别不平衡数据的鲁棒性

### 数据覆盖
- **边保留率**：100%（23,077/23,077）
- **用户覆盖率**：100%（20,001/20,001）
- **游戏覆盖率**：100%（734/734）
- **稀疏度**：0.1572%（推荐系统的健康水平）

## 🎨 可视化示例

运行可视化脚本生成图表：

```python
python visualize_results.py
```

将创建：
1. **性能概览**：AUC/PR-AUC分数和图统计
2. **边权重分析**：分布和改进指标
3. **特征分析**：用户/游戏特征分布和相关性
4. **图结构**：网络拓扑和连接模式
5. **总结信息图**：完整系统概览

## 🔮 未来改进方向

### 模型增强
- [ ] 实现图注意力网络（GAT）引入注意力机制
- [ ] 添加时间动态特征实现动态推荐
- [ ] 多任务学习（评分预测+推荐）
- [ ] 冷启动问题处理（新用户/新游戏）

### 特征工程
- [ ] 添加游戏类型信息
- [ ] 纳入用户人口统计数据
- [ ] 整合社交网络特征
- [ ] 时间序列特征（游戏模式随时间变化）

### 系统优化
- [ ] 小批次训练支持更大数据集
- [ ] 模型压缩和量化
- [ ] 分布式训练支持
- [ ] 实时推理优化

## 📝 引用

如果您在研究中使用此代码，请引用：

```bibtex
@software{steam_gnn_recommendation,
  title = {Steam GNN Recommendation System},
  author = {Yu Xiaolin},
  year = {2025},
  url = {https://github.com/yourusername/steam-gnn-recommendation}
}
```

## 🤝 贡献

欢迎贡献！请随时提交Pull Request。

1. Fork本仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **PyTorch Geometric** 团队提供的优秀GNN库
- **Steam** 提供评论数据平台
- **Hugging Face** 提供transformers库用于主题分析
- **VADER** 情感分析工具

## 📧 联系方式

如有问题或反馈，请在GitHub上提交issue。

---

**用 ❤️ 为Steam游戏社区打造**

*最后更新：2025年10月*
