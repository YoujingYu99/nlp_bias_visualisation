import pandas as pd
import plotly.express as px


token_list = ['I', 'like', 'home', 'nurse', 'mathematician']
values_list = [0, -3, -6, -1, 7]


def bar_cont(token_list, value_list):
    df = pd.DataFrame([(token_list, value_list) for token_list, value_list in zip(token_list, value_list)])

    fig = px.scatter(df, x="words", y="bias_values", color="bias_values",
                     color_continuous_scale=["red", "green", "blue"])

    fig.show()


bar_cont(token_list, values_list)