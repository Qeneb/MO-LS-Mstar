import json
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 30  # 设置字体大小
plt.rcParams['figure.figsize'] = (30, 30)
with open("exp8_scen2_objs.json", "r") as f:
    obj_dic = json.load(f)
x = []
y = []
for id in obj_dic:
    tmp_matrix = obj_dic[id]
    x.append(sum(tmp_matrix[0]))
    y.append(sum(tmp_matrix[1]))

plt.plot(x,y,'.')
plt.show()