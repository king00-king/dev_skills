# 文本分类项目

## 核心功能说明

本项目实现了一个基于传统机器学习方法的文本分类器，支持**高频词**和**TF-IDF**两种特征提取模式，核心功能如下：

1. **文本预处理**  
   对输入文本进行清洗、分词和停用词过滤，示例代码如下：
   ```python
   def preprocess(text):
       text = re.sub(r'[^\w\s]', '', text)  # 去除标点
       words = jieba.cut(text)               # 分词
       words = [w for w in words if w not in stopwords]  # 过滤停用词
       return words
   ```

2. **特征提取**  
   - **高频词模式**：选择语料中出现频率最高的前N个词作为特征，特征向量为词频统计  
     $$ \text{Feature}_i = \text{Count}(w_i) $$
   - **TF-IDF模式**：计算词频与逆文档频率的乘积作为特征权重  
     $$ \text{TF-IDF}(t, d) = \underbrace{\text{TF}(t, d)}_{\text{词频}} \times \underbrace{\log \frac{N}{\text{DF}(t) + 1}}_{\text{逆文档频率}} $$

3. **分类与评估**  
   使用朴素贝叶斯分类器进行训练，并输出准确率与F1-score：
   ```python
   model = MultinomialNB()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
   ```

---

## 特征模式切换方法

### 方法1：通过配置文件修改
在 `config.ini` 中设置 `feature_method` 参数：
```ini
[Feature]
feature_method = tfidf  # 可选值：freq 或 tfidf
top_k = 1000            # 高频词模式下使用的特征维度
```

### 方法2：通过命令行参数指定
运行脚本时直接传入参数：
```bash
python train.py --feature_method freq --top_k 500
```

### 方法3：代码内动态切换
在初始化特征提取器时指定模式：
```python
from feature_extractor import FeatureExtractor

# 高频词模式
extractor = FeatureExtractor(method='freq', top_k=1000)

# TF-IDF模式
extractor = FeatureExtractor(method='tfidf')
```
完成第五点和第六点的代码修改如下：

```python
import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words

def get_top_words(filenames, top_num):
    """从指定文件列表中提取高频词"""
    all_words = []
    for filename in filenames:
        all_words.append(get_words(filename))
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

def build_feature_vector(filenames, top_words):
    """构建特征向量"""
    vectors = []
    for filename in filenames:
        words = get_words(filename)
        vectors.append([words.count(word) for word in top_words])
    return np.array(vectors)

# 初始化数据
all_files = ['邮件_files/{}.txt'.format(i) for i in range(151)]
labels = np.array([1]*127 + [0]*24)  # 标签分布

# 划分训练集和测试集
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    range(len(all_files)), labels, 
    test_size=0.2, 
    stratify=labels,
    random_state=42
)

# 提取训练集特征
train_files = [all_files[i] for i in X_train_idx]
top_words = get_top_words(train_files, 100)
X_train = build_feature_vector(train_files, top_words)

# 样本平衡处理（SMOTE过采样）
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# 训练模型
model = MultinomialNB()
model.fit(X_resampled, y_resampled)

# 测试集评估
test_files = [all_files[i] for i in X_test_idx]
X_test = build_feature_vector(test_files, top_words)
y_pred = model.predict(X_test)

print("=== 分类评估报告 ===")
print(classification_report(
    y_test, 
    y_pred,
    target_names=["普通邮件", "垃圾邮件"],
    digits=4
))

# 预测新邮件的函数
def predict(filename):
    """对未知邮件分类"""
    words = get_words(filename)
    current_vector = np.array([words.count(word) for word in top_words]).reshape(1, -1)
    result = model.predict(current_vector)
    return '垃圾邮件' if result == 1 else '普通邮件'

# 测试预测功能
test_files = ['邮件_files/{}.txt'.format(i) for i in range(151, 156)]
for file in test_files:
    print(f'{file} 分类情况: {predict(file)}')
```

主要修改点说明：

1. **数据集划分**：
   ```python
   X_train_idx, X_test_idx, y_train, y_test = train_test_split(...)
   ```
   使用分层抽样保持类别比例，20%数据作为测试集

2. **样本平衡处理**：
   ```python
   sm = SMOTE(random_state=42)
   X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
   ```
   在训练集上应用SMOTE算法进行过采样

3. **评估指标增强**：
   ```python
   print(classification_report(...))
   ```
   输出包含精度/召回率/F1值的详细报告

4. **特征提取优化**：
   ```python
   def build_feature_vector(...)
   ```
   独立特征构建函数，确保训练/测试集使用相同的特征空间

5. **数据泄漏防护**：
   ```python
   top_words = get_top_words(train_files, 100)
   ```
   仅使用训练集数据生成高频词特征，避免测试集信息泄露

输出示例：
```
=== 分类评估报告 ===
              precision    recall  f1-score   support

        普通邮件     0.3333    0.8000    0.4706         5
        垃圾邮件     0.9474    0.6923    0.8000        26

    accuracy                         0.7097        31
   macro avg     0.6404    0.7462    0.6353        31
weighted avg     0.8483    0.7097    0.7469        31

邮件_files/151.txt 分类情况: 普通邮件
邮件_files/152.txt 分类情况: 垃圾邮件
邮件_files/153.txt 分类情况: 普通邮件
邮件_files/154.txt 分类情况: 垃圾邮件
邮件_files/155.txt 分类情况: 普通邮件

```
<img src=>
