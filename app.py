# https://blog.amedama.jp/entry/streamlit-tutorial

import datetime
import logging

import pandas as pd
import numpy as np
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
# from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

for p in ['sklaern']:
	logging.getLogger(p).setLevel(logging.WARNING)


RANGE_COL = {
	'Population': float,
	}

SINGLE_COL = {
	}

MULTI_COL = {
	'HouseAge': int,
	}

SELECT_ALL = '(no filter)'
N_PLACEHOLDER = 3
MAX_TABLE_ROWS = 18


def filter_data(df, filter_dict):
	for filter_key, filter_value in filter_dict.items():
		if filter_value == SELECT_ALL:
			continue

		if filter_key in RANGE_COL.keys():
			try:
				df = df[df[filter_key].apply(lambda x: filter_value[0] <= x <= filter_value[1])]
			except Exception as e:
				logger.error(filter_key, filter_value, e)
				continue
		elif filter_key in SINGLE_COL.keys():
			try:
				df = df[df[filter_key] == filter_value]
			except Exception as e:
				logger.error(filter_key, filter_value, e)
				continue
		elif filter_key in MULTI_COL.keys():
			try:
				df = df[df[filter_key].apply(lambda x: x in filter_value)]
			except Exception as e:
				logger.error(filter_key, filter_value, e)
				continue

	return df


def set_slider(df, filter_key, dtype=datetime.date, placeholder=None):
	if placeholder is None:
		placeholder = st

	if dtype in (float, int):
		value_range = [float(df[filter_key].min()), float(df[filter_key].max())]
		selected_range = placeholder.slider(filter_key, min_value=value_range[0], max_value=value_range[1], value=value_range)
	elif dtype == datetime.date:
		value_range = [df[filter_key].min(), df[filter_key].max()]
		selected_range = placeholder.date_input(filter_key, value_range)
	else:
		selected_range = list()
	return selected_range


def set_single_select(df, filter_key, dtype=str, placeholder=None):
	if placeholder is None:
		placeholder = st

	if dtype == str:
		filter_value = [SELECT_ALL] + list(df[filter_key].unique())
		selected_item = placeholder.selectbox(filter_key, filter_value)
	else:
		selected_item = None
	return selected_item


def set_multi_select(df, filter_key, dtype=str, placeholder=None):
	if placeholder is None:
		placeholder = st

	if dtype in (str, int):
		filter_value = list(df[filter_key].unique())
		selected_items = placeholder.multiselect(label=filter_key, options=filter_value, default=filter_value)
	else:
		selected_items = list()
	return selected_items


def draw_line(df_, x_col, y_col, title, discrete_digit=None, is_ratio=False, hue=None, hue_filter_values=None, placeholder=None):
	df = df_.copy()
	categories = np.sort(df[hue].unique()).tolist()

	if discrete_digit is not None:
		x_discrete = 'discrete_{}'.format(x_col)
		df[x_discrete] = df[x_col].apply(lambda x: round(x, discrete_digit))
		df = df.groupby([x_discrete, hue]).sum()[y_col].unstack().fillna(0)
	else:
		df = df.groupby([x_col, hue]).sum()[y_col].unstack().fillna(0)

	if is_ratio:
		df['total'] = df.sum(axis=1)
		for c in categories:
			df[c] = df[c] / df['total']
		df.drop('total', axis=1, inplace=True)

	if placeholder is None:
		placeholder = st

	placeholder.write(title)

	if hue_filter_values is not None:
		df = df[[c for c in df.columns if c in hue_filter_values]]

	cols = list()
	for c in df.columns:
		if type(c) == str:
			cols.append(c)
		elif type(c) == int:
			cols.append('{:02d}'.format(c))
		else:
			cols.append(str(c))
	df.columns = cols
	placeholder.line_chart(df)


def draw_bar(df_, x_col, y_col, title, discrete_digit=None, is_ratio=False, hue=None, hue_filter_values=None, placeholder=None):
	df = df_.copy()
	categories = np.sort(df[hue].unique()).tolist()

	if discrete_digit is not None:
		x_discrete = 'discrete_{}'.format(x_col)
		df[x_discrete] = df[x_col].apply(lambda x: round(x, discrete_digit))
		df = df.groupby([x_discrete, hue]).sum()[y_col].unstack().fillna(0)
	else:
		df = df.groupby([x_col, hue]).sum()[y_col].unstack().fillna(0)

	if is_ratio:
		df['total'] = df.sum(axis=1)
		for c in categories:
			df[c] = df[c] / df['total']
		df.drop('total', axis=1, inplace=True)

	if placeholder is None:
		placeholder = st

	placeholder.write(title)

	if hue_filter_values is not None:
		df = df[[c for c in df.columns if c in hue_filter_values]]

	cols = list()
	for c in df.columns:
		if type(c) == str:
			cols.append(c)
		elif type(c) == int:
			cols.append('{:02d}'.format(c))
		else:
			cols.append(str(c))
	df.columns = cols
	placeholder.bar_chart(df)


def draw_hist(df_, col, bins, title, hue=None, hue_filter_values=None, placeholder=None):
	df = df_.copy()
	bin_size = 1 / bins

	if hue is None:
		hue = ['count']

	fig_hist = ff.create_distplot(hist_data=df[[col]].values.reshape(1, -1), group_labels=hue, bin_size=bin_size, show_rug=False)
	fig_hist.update_layout(
		width=500, height=350,
		margin=dict(l=20, r=20, t=30, b=20),
		title_text=title,
		font=dict(size=12),
		)

	if placeholder is None:
		placeholder = st

	placeholder.plotly_chart(fig_hist)


def show_table(df_, placeholder=None):
	df = df_.copy()

	if placeholder is None:
		placeholder = st

	selected_items = placeholder.multiselect(label='Select columns', options=df.columns, default=list())
	if len(selected_items) > 0:
		if placeholder.button('Descending'):
			df = df.sort_values(by=selected_items, ascending=False)
		else:
			df = df.sort_values(by=selected_items, ascending=True)

	prioritized_cols = [c for c in list(RANGE_COL.keys()) + list(SINGLE_COL.keys()) + list(MULTI_COL.keys()) if c not in selected_items]
	rest_cols = [c for c in df.columns if c not in (prioritized_cols + selected_items)]

	placeholder.table(df[selected_items + prioritized_cols + rest_cols].head(MAX_TABLE_ROWS))


def main():
	housing = fetch_california_housing()
	df_X = pd.DataFrame(housing.data, columns=housing.feature_names)
	df_y = pd.DataFrame(housing.target, columns=housing.target_names)
	df = pd.concat([df_X, df_y], axis=1)

	st.set_page_config(page_title="Data Science with Streamlit", layout="wide")

	# データ自体をフィルタする
	st.sidebar.title('Data Filter')

	filter_dict = dict()

	for filter_key, dtype in RANGE_COL.items():
		selected_range = set_slider(df, filter_key, dtype, st.sidebar)
		filter_dict.update({filter_key: selected_range})

	for filter_key, dtype in SINGLE_COL.items():
		selected_item = set_single_select(df, filter_key, dtype, st.sidebar)
		filter_dict.update({filter_key: selected_item})

	for filter_key, dtype in MULTI_COL.items():
		selected_item = set_multi_select(df, filter_key, dtype, st.sidebar)
		filter_dict.update({filter_key: selected_item})

	# データフィルタの確認
	st.sidebar.write(filter_dict)

	# データフィルタを適用したデータフレームを取得
	df = filter_data(df, filter_dict)

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
