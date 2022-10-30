import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入数据
TRAIN_PATH = './data/train_dataset/'
TEST_PATH = './data/test_dataset/'
train_df = pd.read_csv(TRAIN_PATH + 'nCoV_100k_train.labled.csv', engine='python')
test_df = pd.read_csv(TEST_PATH + 'nCov_10k_test.csv', engine='python')

# 观察训练和测试数据的前几行
# print(train_df.head(5))
# print(test_df.head(5))

# 数据的字段包含微博id，微博发布时间，微博中文内容，微博图片，微博视频 和情感倾向(标签)
# print(train_df.shape, test_df.shape)  # ((100000, 7), (10000, 6))

# 100k条带标注的训练数据，10k测试数据
# 输出情感倾向为'-'的数据
# print(train_df[train_df['情感倾向'] == '-'])

# 筛选前图像
train_df['情感倾向'].value_counts().plot.bar()
plt.title('sentiment(target)')
# plt.show()

# 选取出情感倾向为0，1，-1的数据
train_df = train_df[train_df['情感倾向'].isin(['0', '1', '-1'])]

# 筛选后图像
train_df['情感倾向'].value_counts().plot.bar()
plt.title('sentiment(target)')

# 观察舆情趋势与时间的关系，时间轴为2020年1-1日起的自然日
train_df['time'] = pd.to_datetime('2020年' + train_df['微博发布时间'],
                                  format='%Y年%m月%d日 %H:%M', errors='ignore')
train_df['month'] = train_df['time'].dt.month
train_df['day'] = train_df['time'].dt.day
train_df['dayfromzero'] = (train_df['month'] - 1) * 31 + train_df['day']

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.kdeplot(train_df.loc[train_df['情感倾向'] == '0', 'dayfromzero'], ax=ax[0], label='sent(0)')
sns.kdeplot(train_df.loc[train_df['情感倾向'] == '1', 'dayfromzero'], ax=ax[0], label='sent(1)')
sns.kdeplot(train_df.loc[train_df['情感倾向'] == '-1', 'dayfromzero'], ax=ax[0], label='sent(-1)')

train_df.loc[train_df['情感倾向'] == '0', 'dayfromzero'].hist(ax=ax[1])
train_df.loc[train_df['情感倾向'] == '1', 'dayfromzero'].hist(ax=ax[1])
train_df.loc[train_df['情感倾向'] == '-1', 'dayfromzero'].hist(ax=ax[1])

ax[1].legend(['sent(0)', 'sent(1)', 'sent(-1)'])

plt.show()

# 舆情在春节期间迅速升温，并在李文亮医生事件后达到巅峰(2-8日到2月10日)
# 重复发帖数量前10的微博id
# print(train_df['微博id'].value_counts().head(10))

# 每篇微博长度分析
train_df['weibo_len'] = train_df['微博中文内容'].astype(str).apply(len)

sns.kdeplot(train_df['weibo_len'])
plt.title('weibo_len')
plt.show()

# eval()函数用来执行一个字符串表达式，并返回表达式的值；len()方法返回对象（字符、列表、元组等）长度或项目个数
# train_df['pic_len']为图片的张数组成的列表
train_df['pic_len'] = train_df['微博图片'].apply(lambda x: len(eval(x)))
train_df['pic_len'].value_counts().plot.bar()
plt.title('pic_len(target)')
plt.show()

# 统计分析不同图片数量微博的情感倾向
sns.countplot(x='pic_len', hue='情感倾向', data=train_df)
plt.show()
