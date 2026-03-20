"""
算法: 基于物品的协同过滤
场景: 隐式反馈场景(无评分)
版本: 二元版本(例如只知道用户对物品的点击行为)
算法: Jacard相似系数
相似度 = 同时喜欢A和B的个数 / 喜欢A或喜欢B的个数
"""
from collections import defaultdict

class ItemCFBinaryJaccard:
    def __init__(self):
        """
        user_items
            用途: 每个用户交互过的物品
            示例: {'A': {'手机', '耳机'}, 'B': {'手机', '充电宝'}, 'C': {'键盘', '耳机', '充电宝'}, 'D': {'手机'}, 'E': {'键盘'}}
        item_users
            用途: 每个物品有哪些用户交互过
            示例: {'手机': {'A', 'D', 'B'}, '耳机': {'C', 'A'}, '充电宝': {'C', 'B'}, '键盘': {'C', 'E'}}
        item_item_sim
            用途: 物品之间的相似度
            示例: {'手机': {'耳机': 0.25, '充电宝': 0.25, '键盘': 0.0}, '耳机': {'手机': 0.25, '充电宝': 0.33, '键盘': 0.33}}
        """
        self.user_items=defaultdict(set)
        self.item_users=defaultdict(set)
        self.item_item_sim=defaultdict(dict)
    
    def fit(self, datas):
        """
        拟合数据(训练)

        Args:
            datas (list): 数据集，[('A', '手机'), ('A', '耳机')]
        """
        for user, item in datas:
            self.user_items[user].add(item)
            self.item_users[item].add(user)

        for item, users in self.item_users.items():
            for item2, users2 in self.item_users.items():
                if item != item2 and not self.item_item_sim[item].get(item2):
                    sim = len(users & users2) / len(users | users2)
                    self.item_item_sim[item][item2] = sim
                    self.item_item_sim[item2][item] = sim

    def similar(self, item1, item2):
        """获取两个物品之间的相似度"""
        return self.item_item_sim[item1][item2]
    
    def recommend(self, user, top_n):
        """
        给用户推荐物品

        Args:
            user (str):     用户ID
            top_n (int):    推荐数量

        Returns:
            list: 推荐物品列表, [('充电宝', 0.58), ('键盘', 0.33)]
        """
        # 1. 存储推荐物品
        recommends={}
        # 2. 获取用户交互过的物品
        items=self.user_items[user]
        # 3. 遍历交互过的物品
        for item in items:
            # 获取每个物品与其他物品的相似度
            for item2, sim in self.item_item_sim[item].items():
                # 只有当物品没有被用户交互过且相似度>0才添加到备选列表
                if item2 not in items and sim > 0:
                    sim_ = recommends.get(item2, 0)
                    recommends[item2] = sim_ + sim
        # 4. 对备选列表进行排序并取TopN
        return sorted(recommends.items(), key=lambda x: x[1], reverse=True)[:top_n]


if __name__=="__main__":
    # 1. 模拟数据
    datas=[
        ('A', '手机'),
        ('A', '耳机'),
        ('B', '手机'),
        ('B', '充电宝'),
        ('C', '耳机'),
        ('C', '充电宝'),
        ('C', '键盘'),
        ('D', '手机'),
        ('E', '键盘')
    ]

    # 2. 算法拟合
    itemcf = ItemCFBinaryJaccard()
    itemcf.fit(datas)

    # 3. 推荐测试
    print(f"给用户 A 推荐的物品: {itemcf.recommend('A', 3)}")
    print(f"给用户 B 推荐的物品: {itemcf.recommend('B', 3)}")
    print(f"给用户 C 推荐的物品: {itemcf.recommend('C', 3)}")
    print(f"给用户 D 推荐的物品: {itemcf.recommend('D', 3)}")
    print(f"给用户 E 推荐的物品: {itemcf.recommend('E', 3)}")
    print(f"给用户 F 推荐的物品: {itemcf.recommend('F', 3)}")