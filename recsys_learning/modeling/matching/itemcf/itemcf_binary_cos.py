"""
算法: 基于物品的协同过滤
场景: 隐式反馈场景(无评分)
版本: 二元版本(例如只知道用户对物品的点击行为)
算法: 余弦相似度
"""

from collections import defaultdict
import math

class ItemCFBinaryCos:
    def __init__(self):
        self.item_users=defaultdict(set)
        self.user_items=defaultdict(set)
        self.item_item_sim=defaultdict(dict)

    def fit(self, datas):
        for user, item in datas:
            self.item_users[item].add(user)
            self.user_items[user].add(item)
        
        for item1, users1 in self.item_users.items():
            for item2, user2 in self.item_users.items():
                if item1==item2:
                    continue
                sim = len(users1 & user2) / math.sqrt(len(users1) * len(user2))
                self.item_item_sim[item1][item2] = sim
                self.item_item_sim[item2][item1] = sim

    def recommend(self, user, top_n):
        recommends = {}
        historys = self.user_items[user]
        for history in historys:
            for item, sim in self.item_item_sim[history].items():
                if item in historys or sim <= 0:
                    continue
                sim_ = recommends.get(item, 0)
                recommends[item] = sim_+sim
        
        return sorted(recommends.items(), key=lambda x:x[1], reverse=True)[:top_n]

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

    # 2. 训练模型(数据拟合)
    itemcf = ItemCFBinaryCos()
    itemcf.fit(datas)

    # 3. 推荐测试
    print(f"给用户 A 推荐的物品: {itemcf.recommend('A', 3)}")
    print(f"给用户 B 推荐的物品: {itemcf.recommend('B', 3)}")
    print(f"给用户 C 推荐的物品: {itemcf.recommend('C', 3)}")
    print(f"给用户 D 推荐的物品: {itemcf.recommend('D', 3)}")
    print(f"给用户 E 推荐的物品: {itemcf.recommend('E', 3)}")
    print(f"给用户 F 推荐的物品: {itemcf.recommend('F', 3)}")
