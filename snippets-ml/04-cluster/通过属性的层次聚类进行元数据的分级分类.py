import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


path = 'D:/sinux/doc/产品-数据资产管理方向/AI标准辅助/参考文件/输出/附录F-设备类元数据.xlsx'

data = pd.read_excel(path, '对象类', header=1)
print(data)

data.dropna(subset=['对象类编码', '分类编码'], inplace=True)

# 编码到名称的映射
code_to_name = dict(data[['对象类编码', '中文名称']].values)
p2c = dict(data[['分类编码', '对象类编码']].values)
c2p = dict(data[['对象类编码', '分类编码']].values)


def find_label_path(code: str):
    if code == 'COMP':
        return '/设备'
    else:
        return find_label_path(c2p[code]) + f'/{code_to_name[code]}'


data['label'] = data['对象类编码'].apply(find_label_path)
train_data = pd.DataFrame({
    'code': data['对象类编码'].values,
    'label': data['对象类编码'].apply(find_label_path).values,
})
print(train_data)
properties = pd.read_excel(path, '对象类属性', header=0)
properties = properties[properties['状态'] == '有效']
properties.dropna(subset=['对象类编码', '属性编码'], inplace=True)
code_and_property = {(row['对象类编码'], row['属性名称']) for _, row in properties.iterrows()}

statistics = []
for property_name in set(properties['属性名称']):
    s = [1 if (each, property_name) in code_and_property else 0 for each in train_data['code'].values]
    statistics.append((property_name, sum(s)))
    train_data[property_name] = s

statistics.sort(key=lambda a: -a[-1])
# 解包 statistics 列表，获取属性名称和出现次数
attributes, counts = zip(*statistics)

# # 创建直方图
# plt.figure(figsize=(10, 6))
# plt.bar(attributes, counts, color='skyblue')
# # 添加标题和标签
# plt.title('属性出现次数直方图')
# plt.xlabel('属性')
# plt.ylabel('出现次数')
# # 旋转 x 轴标签，避免重叠
# plt.xticks(rotation=45)
# # 由于数据量很大, 超过1000各, 每隔150个展示一个刻度
# tick_spacing = 150
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
# statistics_df = pd.DataFrame({'属性': [each[0] for each in statistics], '计数': [each[1] for each in statistics]})
# statistics_df.to_excel('statistics.xlsx')
# # 显示图表
# plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
# # 显示图表
# plt.show()

# 选择属性列
attributes = [each[0] for each in statistics]

# 使用层次聚类
linked = linkage(train_data[attributes], 'single')

tree = to_tree(linked)


def merge_nodes(node, merged_nodes):
    if node.is_leaf():
        return [node.id]
    else:
        left_leaves = merge_nodes(node.left, merged_nodes)
        right_leaves = merge_nodes(node.right, merged_nodes)

        # 如果距离为0，合并左右子节点
        if node.dist == 0:
            merged_nodes.append((left_leaves + right_leaves, node.id))
            return []
        else:
            return left_leaves + right_leaves


# 存储合并的节点
merged_nodes = []
# merge_nodes(tree, merged_nodes)


# 重新构建树结构
def build_tree(node, merged_nodes):
    if node.is_leaf():
        return {'id': node.id, 'is_leaf': True, 'label': train_data['label'].iloc[node.id], 'leaves': [node.id]}
    else:
        left_tree = build_tree(node.left, merged_nodes)
        right_tree = build_tree(node.right, merged_nodes)

        # 如果当前节点距离为0，合并左右子节点
        if node.dist == 0:
            merged_leaves = left_tree['leaves'] + right_tree['leaves']

            return {'id': node.id, 'is_leaf': False, 'label': f"{left_tree['label'], right_tree['label']}", 'leaves': merged_leaves}
        else:
            return {
                'id': node.id,
                'is_leaf': False,
                'label': f"节点 {node.id} (距离: {node.dist:.2f})",
                'left': left_tree,
                'right': right_tree,
                'leaves': left_tree['leaves'] + right_tree['leaves']
            }


# 重新构建树结构
# merged_tree = build_tree(tree, merged_nodes)


def print_tree(node, prefix=""):
    if node.is_leaf():
        print(f"{prefix}对象类: {train_data['label'].iloc[node.id]}")
    else:
        print(f"{prefix}节点: {node.id} (距离: {node.dist:.2f})")
        print_tree(node.left, prefix + "  ├─ ")
        print_tree(node.right, prefix + "  └─ ")


# 打印树结构
print_tree(tree)

# 打印树结构
# print_tree(merged_tree)


# # 绘制树状图
# plt.figure(figsize=(10, 7))
# dendrogram(linked, orientation='top', labels=train_data['label'].values, distance_sort='descending', show_leaf_counts=True)
# plt.title('层次聚类树状图')
# plt.xlabel('对象类')
# plt.ylabel('距离')
# plt.show()
