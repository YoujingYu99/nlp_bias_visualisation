import pandas as pd
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


token_list = ['I', 'like', 'home', 'nurse', 'mathematician']
value_list = [0, -3, -6, -1, 7]

# df = pd.DataFrame([(token_list, value_list) for token_list, value_list in zip(token_list, value_list)])
# print(df)
#
# plt.style.use('ggplot')
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# fig, (ax1, ax2) = plt.subplots(2)
#
# ax.bar(token_list,
#            value_list,
#            yerr=df.std(axis=1) / np.sqrt(len(df.columns)),
#            color=colors)
#
# norm = mpl.colors.Normalize(vmin=0, vmax=1)
#
# mpl.colorbar.ColorbarBase(ax2, cmap=cm.RdBu,
#                                 norm=norm,
#                                 orientation='horizontal')
#
# plt.show()



def draw_bar(key_name, key_values):
    plt.rcParams['axes.unicode_minus'] = False


    def autolable(rects):
        for rect in rects:
            height = rect.get_height()
            if height >= 0:
                plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height + 0.02, '%.3f' % height)
            else:
                plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height - 0.06, '%.3f' % height)
                plt.axhline(y=0, color='black')

    # normalise
    norm = plt.Normalize(-1, 1)
    norm_values = norm(key_values)
    map_vir = cm.get_cmap(name='viridis')
    colors = map_vir(norm_values)
    fig = plt.figure()  # 调用figure创建一个绘图对象
    plt.subplot(111)
    ax = plt.bar(key_name, key_values, width=0.5, color=colors, edgecolor='black')  # edgecolor边框颜色

    sm = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
    sm.set_array([])
    plt.colorbar(sm)
    autolable(ax)

    plt.show()

draw_bar(token_list, value_list)





# def bar_cont(token_list, value_list):
#     df = pd.DataFrame([(token_list, value_list) for token_list, value_list in zip(token_list, value_list)])
#
#     fig = px.scatter(df, x="words", y="bias_values", color="bias_values",
#                      color_continuous_scale=["red", "green", "blue"])
#
#     fig.show()
#
#
# bar_cont(token_list, values_list)