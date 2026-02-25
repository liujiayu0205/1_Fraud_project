# -*- coding: utf-8 -*-
"""
信用卡欺诈检测 - H2O AutoML 建模（改进版）
基于时间序列感知的交叉验证，去除易过拟合特征，增加时间特征
数据集：fraudTrain.csv（训练集）、fraudTest.csv（测试集）
要求：Python 3.6+，安装 h2o, pandas, numpy
"""

import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML

# -------------------- 1. 文件路径设置 --------------------
TRAIN_PATH = "./dataset/fraudTrain.csv"
TEST_PATH = "./dataset/fraudTest.csv"

# -------------------- 2. 读取数据 --------------------
print("读取训练集...")
train_df = pd.read_csv(TRAIN_PATH)
print(f"训练集形状：{train_df.shape}")
print("读取测试集...")
test_df = pd.read_csv(TEST_PATH)
print(f"测试集形状：{test_df.shape}")

# -------------------- 3. 数据预处理与特征工程 --------------------
target = 'is_fraud'

# 3.1 删除明显易过拟合的特征（标识符、卡号、姓名、地址等）
drop_cols = [
    'Unnamed: 0',       # 行号，完全无意义
    'trans_num',        # 交易唯一ID，会导致严重过拟合
    'cc_num',           # 信用卡号，测试集中新卡号无法泛化
    'first',            # 名字
    'last',             # 姓氏
    'street',           # 街道地址
    'zip',              # 邮政编码（可能与城市冗余）
    'dob',              # 出生日期（将转换为年龄）
    'trans_date_trans_time'  # 原始时间戳，将提取特征后删除
]
# 仅删除存在的列
drop_cols = [col for col in drop_cols if col in train_df.columns]

print(f"删除的列：{drop_cols}")
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

# 3.2 处理出生日期 dob（已删除，但如果未删除可保留年龄转换）
# 此处假设 dob 已被删除，若未删除可取消注释以下代码
# 但我们已经删除了 dob，所以无需处理

# 3.3 从交易时间提取特征（原始列已删除，但我们需要重新解析？）
# 注意：原始 trans_date_trans_time 已被删除，但我们仍需时间特征。
# 实际上，应该先提取时间特征再删除原始列。
# 重新读取数据或调整顺序：先提取时间特征，再删除原始列。
# 为了避免重复读取，我们重新加载数据并正确顺序处理。

# 更稳健的做法：重新加载数据，先提取时间特征，再删除无用列。
# 但为了保持代码简洁，我们在此重新执行一次（但实际运行时前面的删除操作已改变数据框）。
# 我们在这里重新执行完整流程：

# 重新读取数据（避免上面删除影响）
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 转换时间列
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'])

# 提取时间特征
for df in [train_df, test_df]:
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek  # 0=周一,6=周日
    df['day_of_month'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# 删除易过拟合特征（包括原始时间戳）
drop_cols = [
    'Unnamed: 0', 'trans_num', 'cc_num', 'first', 'last', 'street', 'zip', 'dob',
    'trans_date_trans_time'
]
drop_cols = [col for col in drop_cols if col in train_df.columns]
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

# 3.4 检查是否有缺失值
print("\n训练集缺失值统计：")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
print("\n测试集缺失值统计：")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# 若存在缺失值，简单填充（此处假设无缺失）
# 但为了安全，仍保留填充代码
for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# 3.5 查看类别分布
print("\n训练集目标变量分布：")
print(train_df[target].value_counts())
print("\n测试集目标变量分布：")
print(test_df[target].value_counts())

# -------------------- 4. 创建时间序列交叉验证折 --------------------
# 目标：在训练集内部按时间顺序划分 5 折，保证每折的训练数据时间早于验证数据。
# 方法：使用交易时间（这里我们已删除原始时间列，但可用 month 等代替？不准确）
# 实际上我们还有 unix_time 列，它是时间戳，可以用它排序。
# 重新加载数据，保留 unix_time 用于排序（我们之前删除了吗？没有删除 unix_time）
# unix_time 还在，可以用来排序。
# 注意：unix_time 是整数时间戳，可以直接用于排序。

# 由于我们重新读取数据后保留了 unix_time，现在用它创建时间折。
# 将训练集按 unix_time 排序
train_df = train_df.sort_values('unix_time').reset_index(drop=True)

# 创建时间折（例如按时间顺序分成5份，每份作为验证集）
n_folds = 5
fold_size = len(train_df) // n_folds
train_df['fold'] = 0
for i in range(n_folds):
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(train_df)
    train_df.loc[start_idx:end_idx-1, 'fold'] = i

# 这样 fold 列的值 0,1,2,3,4 表示按时间顺序的分组，越小的索引时间越早。
# 在交叉验证中，我们将使用 fold 列指定验证集：例如 fold=4 作为验证，fold<4 作为训练。
# 但 H2O 的 fold_column 需要指定一个列，其中的值表示每行属于哪个折，并且会自动进行 k 折，
# 但不能保证时间顺序（默认是随机分配）。我们需要自定义交叉验证：手动进行多次训练。
# 更简单的方法：使用 H2O 的 fold_column，但需要确保 fold 的值按时间顺序，
# 且验证集是最后一个 fold（即时间最晚的 fold）。但 H2O 的交叉验证默认会使用所有 fold 轮流作为验证，
# 这就会导致某些 fold 验证集的时间早于训练集（违反时间顺序）。
# 因此，更好的方法是手动进行多次训练：每次用前 k-1 个 fold 训练，最后一个 fold 验证。
# 但 H2O AutoML 目前不直接支持这种自定义时间序列交叉验证。我们可以采用折中方案：
# 使用单次训练，但设置 validation_frame 为时间上最新的数据（例如最后20%），训练集为前80%。
# 这样模型在训练时就能看到验证集是未来的数据，防止时间泄露。
# 然后 AutoML 内部会使用 training_frame 再做随机交叉验证，但这样仍然存在未来信息进入训练集内部折的风险。
# 为了严格防止时间泄露，我们最好放弃 AutoML 内部的交叉验证，只用单次验证。
# 我们这里采用折中：设置 nfolds=0（不使用交叉验证），而是显式指定 validation_frame。
# 这样 AutoML 将在训练集上训练，并在验证集上评估进行早停和模型选择。
# 这样虽然不能充分利用数据，但能保证时间顺序正确。

# 按时间排序后，取最后20%作为验证集
split_point = int(len(train_df) * 0.8)
train_time_df = train_df.iloc[:split_point].copy()
valid_time_df = train_df.iloc[split_point:].copy()

print(f"时间划分：训练集 {len(train_time_df)} 行，验证集 {len(valid_time_df)} 行")

# -------------------- 5. 初始化 H2O 并加载数据 --------------------
h2o.init(max_mem_size="8G")

train_h2o = h2o.H2OFrame(train_time_df)
valid_h2o = h2o.H2OFrame(valid_time_df)
test_h2o = h2o.H2OFrame(test_df)

# 定义特征与目标
features = [col for col in train_h2o.columns if col != target and col != 'fold' and col != 'unix_time']
# 注意：我们保留了 unix_time 用于排序，但在建模时应该排除，因为它本质是时间戳，可能导致过拟合。
# 另外 fold 列也应该排除。
exclude_cols = ['fold', 'unix_time']
features = [f for f in features if f not in exclude_cols]
print(f"用于建模的特征数量：{len(features)}")
print(f"特征列表：{features}")

y = target

# 将目标列转为因子
train_h2o[y] = train_h2o[y].asfactor()
valid_h2o[y] = valid_h2o[y].asfactor()
test_h2o[y] = test_h2o[y].asfactor()

# -------------------- 6. 运行 H2O AutoML --------------------
print("\n开始 H2O AutoML 训练（时间序列验证）...")
aml = H2OAutoML(
    max_runtime_secs=600,           # 最大运行时间（秒）
    max_models=None,
    seed=42,
    nfolds=0,                        # 不使用交叉验证（已使用自定义时间验证集）
    exclude_algos=None,
    balance_classes=True,            # 保持类别平衡
    class_sampling_factors=None,
    max_after_balance_size=3.0,       # 限制过采样倍数，防止过拟合
    sort_metric="AUC",
    stopping_metric="AUC",
    stopping_rounds=3,
    stopping_tolerance=0.001
)

aml.train(x=features, y=y, training_frame=train_h2o, validation_frame=valid_h2o)

# -------------------- 7. 查看结果 --------------------
lb = aml.leaderboard
print("\nAutoML 排行榜（按 AUC 排序）：")
print(lb.head(rows=lb.nrows))

best_model = aml.leader
print("\n最佳模型：")
print(best_model)

# 在验证集上评估（已由 AutoML 输出），但在测试集上评估
perf_test = best_model.model_performance(test_h2o)
print("\n测试集评估结果：")
print(perf_test)

# 保存模型
model_path = h2o.save_model(model=best_model, path="./models", force=True)
print(f"\n最佳模型已保存至：{model_path}")

# -------------------- 8. 关闭 H2O --------------------
h2o.cluster().shutdown(prompt=False)

print("\n改进版运行完成！")