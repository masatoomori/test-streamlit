# https://blog.amedama.jp/entry/streamlit-tutorial

import logging

import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
# from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

for p in ['sklaern']:
	logging.getLogger(p).setLevel(logging.WARNING)


def main():
	housing = fetch_california_housing()
	df_X = pd.DataFrame(housing.data, columns=housing.feature_names)
	df_y = pd.DataFrame(housing.target, columns=housing.target_names)
	df = pd.concat([df_X, df_y], axis=1)

	st.set_page_config(page_title="Data Science with Streamlit", layout="wide")

	st.write("Hello, world!")

	# スライドバー
	st.sidebar.subheader("スライドバー")
	house_age_min, house_age_max = st.sidebar.slider(label="HouseAge", min_value=0, max_value=60, value=(0, 60))

	# セレクトボックス
	selected_item = st.sidebar.selectbox(label='AveRooms', options=set([int(n) for n in df['AveRooms']]))

	# マルチセレクトボックス
	selected_items = st.sidebar.multiselect(label="HouseAge", options=df['HouseAge'].unique(), default=df['HouseAge'].unique())

	# 選択された値
	st.write("AveRooms: {}".format(selected_item))
	st.write('Selected age: {} - {}'.format(house_age_min, house_age_max))

	# データを絞る
	df = df[df['AveRooms'].apply(lambda x: int(x) == selected_item)]
	df = df[df["HouseAge"].between(house_age_min, house_age_max)]
	df = df[df["HouseAge"].apply(lambda x: x in selected_items)]

	# 折れ線グラフ
	st.subheader("折れ線グラフ")
	st.line_chart(df['HouseAge'])

	# 棒グラフ
	st.subheader("棒グラフ")
	st.bar_chart(df.groupby('HouseAge').mean()['MedHouseVal'])

	# ヒストグラム
	# st.subheader("ヒストグラム")
	# fig = plt.figure(figsize=(20, 4))
	# ax = fig.add_subplot()
	# ax.hist(df['HouseAge'], bins=20)
	# st.pyplot(fig)

	fig_hist = ff.create_distplot(hist_data=df[['HouseAge']].values.reshape(1, -1), group_labels=['HouseAge'], bin_size=1)		# bin_size はいくつに分割するかではなく、いくつのレコードをまとめるか
	fig_hist.update_layout(title_text='Histogram')
	st.plotly_chart(fig_hist)

	# データフレーム
	st.subheader("データフレーム")
	st.dataframe(df.head(100))
	st.table(df.head(100))

	# 地図
	st.subheader("地図")
	df_map = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
	st.map(df_map.head(100))


if __name__ == "__main__":
	main()
