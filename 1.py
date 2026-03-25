import re
import os  # 新增：用于创建文件夹
import jieba
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence


# ====================== 案例1：使用Google预训练Word2Vec模型 ======================
def use_pretrained_model():
    """加载Google预训练模型并实现相似性计算、类比推理"""
    try:
        # 替换为你的预训练模型路径（GoogleNews-vectors-negative300.bin.gz）
        model_path = r"D:\Large Model\worav\GoogleNews-vectors-negative300.bin"  # 加r避免转义
        # 加载预训练模型（二进制格式）
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)

        # 1. 计算两个词的相似性
        word1, word2 = "cat", "dog"
        similarity = model.similarity(word1, word2)
        print(f"【相似性计算】{word1} 和 {word2} 的相似性：{similarity:.4f}")

        # 2. 类比推理：king - man + woman = ?
        result = model.most_similar(positive=["woman", "king"], negative=["man"], topn=5)
        print("\n【类比推理】king - man + woman 的结果：")
        for word, score in result:
            print(f"  {word}: {score:.4f}")

    except FileNotFoundError:
        print("请先下载Google预训练模型，并修改model_path为正确路径！")
    except Exception as e:
        print(f"运行出错：{e}")


# ====================== 案例2：基于《三国演义》训练自定义模型 ======================
def preprocess_text(text_path):
    """文本预处理：清洗+分词"""
    # 1. 读取文本（添加异常捕获，提示更清晰）
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件：{text_path}，请检查文件路径和文件名是否正确！")

    # 2. 文本清洗：去除无关字符、空白符、换行符
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)  # 只保留中文
    text = re.sub(r"\s+", "", text)  # 去除所有空白符

    # 3. 中文分词（jieba），按句子切分（这里按固定长度切分，也可按标点）
    words = jieba.lcut(text)
    # 按每10个词为一个句子（可根据需求调整），过滤空词
    sentences = [words[i:i + 10] for i in range(0, len(words), 10) if words[i:i + 10]]
    return sentences


def train_custom_model():
    """训练《三国演义》Word2Vec模型"""
    # 1. 预处理文本（关键：路径加r前缀，避免转义；同时检查文件名是否正确）
    text_path = r"D:\Large Model\worav\sanguoyanyi.txt" # r前缀：原始字符串，忽略转义字符
    try:
        sentences = preprocess_text(text_path)
        print(f"文本预处理完成，共生成 {len(sentences)} 个句子")
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"文本预处理出错：{e}")
        return

    # 2. 训练Word2Vec模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,  # 词向量维度
        window=5,  # 上下文窗口大小
        min_count=1,  # 最小词频（过滤低频词）
        workers=4,  # 并行训练线程数
        epochs=10  # 训练轮数
    )

    # 3. 保存模型（核心修复：先创建保存目录，再保存模型）
    model_save_dir = r"E:\big model use\model"  # 模型保存目录
    model_save_path = r"E:\big model use\model\sanguo_w2v.model"  # 模型完整路径

    # 检查目录是否存在，不存在则创建
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"创建模型保存目录：{model_save_dir}")

    # 保存模型
    model.save(model_save_path)
    print(f"模型已保存为：{model_save_path}")

    # 4. 模型推理验证
    wv = model.wv  # 获取词向量

    # 4.1 查找与“刘备”最相似的词
    try:
        similar_words = wv.most_similar("刘备", topn=10)
        print("\n【相似词查找】与“刘备”最相似的10个词：")
        for word, score in similar_words:
            print(f"  {word}: {score:.4f}")
    except KeyError:
        print("未找到“刘备”这个词，可能是文本中无该词或分词/清洗时被过滤")

    # 4.2 类比推理：刘备 - 张飞 + 关羽 = ?
    try:
        analogy_result = wv.most_similar(positive=["刘备", "关羽"], negative=["张飞"], topn=5)
        print("\n【类比推理】刘备 - 张飞 + 关羽 的结果：")
        for word, score in analogy_result:
            print(f"  {word}: {score:.4f}")
    except KeyError as e:
        print(f"类比推理出错：缺少词 {e}（可能是分词/清洗时被过滤）")


# ====================== 主函数：运行案例 ======================
if __name__ == "__main__":
    # 运行案例1（需先下载预训练模型）
    # use_pretrained_model()

    # 运行案例2（推荐先运行这个）
    train_custom_model()