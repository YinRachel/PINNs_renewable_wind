import matplotlib.pyplot as plt
from matplotlib import font_manager


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体

# 设置负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 示例数据
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]

# 创建图表
plt.plot(x, y, marker='o')

# 添加标题和标签
plt.title('示例图表')
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 获取当前的图形对象
fig = plt.gcf()

# 添加注释
fig.text(0.5, 0.01, "'这是图表的注释文本，位于图表的底部中央位置'", ha='center')

# 显示图表
plt.show()
