import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import logging

# 设置日志输出，观察训练进度
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ==========================================
# 1. 准备数据 (Data Preparation)
# ==========================================
# Word2Vec 需要的输入格式是：列表的列表 (List of Lists)
# 外层列表代表整个语料库，内层列表代表一个句子（分词后的单词列表）
corpus = [
    ["我", "喜欢", "吃", "苹果"],
    ["苹果", "很", "甜", "而且", "健康"],
    ["香蕉", "也是", "一种", "水果"],
    ["我", "也", "喜欢", "吃", "香蕉"],
    ["深度学习", "和", "机器学习", "都是", "人工智能", "的", "分支"],
    ["机器学习", "需要", "大量", "数据"],
    ["人工智能", "正在", "改变", "世界"],
    ["苹果", "公司", "发布", "了", "新", "手机"],
    ["手机", "和", "电脑", "都是", "电子产品"],
    ["吃", "水果", "对", "身体", "好"]
]

# 如果你的数据是原始字符串，可以使用 simple_preprocess 进行简单分词和清洗
# raw_text = "我喜欢吃苹果。苹果很甜..."
# sentences = [simple_preprocess(doc) for doc in raw_text_list]

print(f"语料库句子数量: {len(corpus)}")
print(f"第一个句子样例: {corpus[0]}")

# ==========================================
# 2. 训练模型 (Model Training)
# ==========================================
# 核心参数说明：
# vector_size: 词向量的维度 (旧版叫 size)，通常设为 100, 200, 300
# window: 上下文窗口大小，即预测目标词时考虑前后多少个词
# min_count: 忽略出现次数少于该值的词 (过滤低频词)
# workers: 使用多少个CPU线程并行训练
# sg: 训练算法 (0=CBOW, 1=Skip-gram)。Skip-gram 对小数据集效果更好，但慢；CBOW 快。
# epochs: 遍历语料库的次数 (旧版叫 iter)

model = Word2Vec(
    sentences=corpus,
    vector_size=100,      # 向量维度
    window=5,             # 上下文窗口
    min_count=1,          # 因为演示数据少，设为1保留所有词；实际大数据可设为5或10
    workers=4,            # 并行线程数
    sg=1,                 # 1 表示 Skip-gram, 0 表示 CBOW
    epochs=5000,            # 训练轮数
    alpha=0.025,          # 初始学习率
    min_alpha=0.0001      # 最小学习率
)

print("\n✅ 模型训练完成！")
print(f"模型参数: {model}")

# ==========================================
# 3. 查看词汇表 (Vocabulary Check)
# ==========================================
# 查看模型学到了哪些词
vocabulary = list(model.wv.index_to_key) # 4.x 版本获取词汇表的新方法
print(f"\n词汇表中的词 ({len(vocabulary)}个): {vocabulary[:10]}...") # 打印前10个

# ==========================================
# 4. 模型应用 (Model Usage)
# ==========================================

# A. 查询某个词的向量
word = "苹果"
if word in model.wv:
    vector = model.wv[word]
    print(f"\n🔹 '{word}' 的向量前5维: {vector[:5]}")
    print(f"   向量总维度: {len(vector)}")
else:
    print(f"\n⚠️ 词汇表中不存在 '{word}'")

# B. 查找最相似的词 (Most Similar)
# 找出与 '机器学习' 最相似的词
target_word = "机器学习"
if target_word in model.wv:
    similar_words = model.wv.most_similar(target_word, topn=3)
    print(f"\n🔹 与 '{target_word}' 最相似的词: {similar_words}")
else:
    print(f"\n⚠️ 词汇表中不存在 '{target_word}'")

# C. 词类比推理 (Word Analogy) - 经典测试
# 公式：国王 - 男人 + 女人 = ? (应该接近 '女王')
# 这里我们用简单的例子：苹果 - 水果 + 电子产品 = ? (可能接近 '手机')
# 注意：由于演示数据极小，类比结果可能不准确，大数据集效果才好
try:
    analogy_result = model.wv.most_similar(positive=['苹果', '电子产品'], negative=['水果'], topn=1)
    print(f"\n🔹 类比推理 ('苹果' - '水果' + '电子产品'): {analogy_result}")
except KeyError as e:
    print(f"\n⚠️ 类比推理失败，缺少词汇: {e}")

# D. 计算两个词的相似度 (Similarity Score)
word1, word2 = "苹果", "香蕉"
if word1 in model.wv and word2 in model.wv:
    similarity = model.wv.similarity(word1, word2)
    print(f"\n🔹 '{word1}' 和 '{word2}' 的余弦相似度: {similarity:.4f} (范围 -1 到 1)")
else:
    print(f"\n⚠️ 无法计算相似度，词汇缺失")

# ==========================================
# 5. 保存与加载模型 (Save & Load)
# ==========================================
# 保存模型 (推荐保存 .kv 格式，只存向量，文件更小；或者存完整模型)
model.save("my_word2vec.model")
model.wv.save("my_word2vec_vectors.kv") # 仅保存向量，用于后续推理，不能继续训练

print("\n💾 模型已保存为 'my_word2vec.model' 和 'my_word2vec_vectors.kv'")

# 加载模型
loaded_model = Word2Vec.load("my_word2vec.model")
loaded_wv = gensim.models.KeyedVectors.load("my_word2vec_vectors.kv")

# 验证加载是否成功
print(f"加载后验证 - '人工智能' 的相似词: {loaded_wv.most_similar('人工智能', topn=2)}")