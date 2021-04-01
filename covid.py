#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import threading

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import arima
from pmdarima.model_selection import train_test_split
from sklearn.metrics import r2_score


# 准备数据
def adjust_date(s):
    t = s.split("/")
    return f"20{t[2]}-{int(t[0]):02d}-{int(t[1]):02d}"


df = pd.read_csv(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
).drop(columns=["Lat", "Long"]).groupby("Country/Region").sum().transpose()
df.index = pd.DatetimeIndex(df.index.map(adjust_date))

data_Brazil = df["Brazil"].diff().dropna()
data_China = df["China"].diff().dropna()
data_France = df["France"].diff().dropna()
data_Germany = df["Germany"].diff().dropna()
data_India = df["India"].diff().dropna()
data_Italy = df["Italy"].diff().dropna()
data_Japan = df["Japan"].diff().dropna()
data_Russia = df["Russia"].diff().dropna()
data_Spain = df["Spain"].diff().dropna()
data_UK = df["United Kingdom"].diff().dropna()
data_US = df["US"].diff().dropna()


def draw(data, name):
    # 模型训练
    model = arima.AutoARIMA(start_p=1,
                            max_p=7,
                            d=1,
                            start_q=0,
                            max_q=2,
                            start_P=1,
                            max_P=7,
                            D=1,
                            start_Q=0,
                            max_Q=2,
                            m=7,
                            seasonal=True,
                            test="adf",
                            trace=True,
                            error_action="ignore",
                            suppress_warnings=True,
                            stepwise=True)
    model.fit(data)

    # 模型验证
    train, test = train_test_split(data, train_size=0.8)
    pred_test = model.predict_in_sample(start=train.shape[0], dynamic=False)
    validating = pd.Series(pred_test, index=test.index)
    r2 = r2_score(test, pred_test)

    # 预测未来
    pred, pred_ci = model.predict(n_periods=14, return_conf_int=True)
    idx = pd.date_range(data.index.max() + pd.Timedelta("1D"), periods=14, freq="D")
    forecasting = pd.Series(pred, index=idx)

    # 绘图呈现
    plt.figure(figsize=(15, 6))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.legend()
    plt.title(f"Daily Increasement Forecasting - {name} (R2 = {r2:.6f})")

    plt.plot(data.index, data, label="实际值", color="blue")
    plt.plot(validating.index, validating, label="校验值", color="orange")
    plt.plot(forecasting.index, forecasting, label="预测值", color="red")
    # plt.fill_between(forecasting.index, pred_ci[:, 0], pred_ci[:, 1], color="black", alpha=.25)

    plt.savefig(os.path.join("figures", f"covid-{name}.png"), bbox_inches="tight")


if __name__ == "__main__":
    threads = []
    for name in [
        (data_Brazil, "Brazil"),
        (data_China, "China"),
        (data_France, "France"),
        (data_Germany, "Germany"),
        (data_India, "India"),
        (data_Italy, "Italy"),
        (data_Japan, "Japan"),
        (data_Russia, "Russia"),
        (data_Spain, "Spain"),
        (data_UK, "UK"),
        (data_US, "US"),
    ]:
        t = threading.Thread(target=draw, args=(name[0], name[1]))
        threads.append(t)

    for t in threads:
        t.start()
