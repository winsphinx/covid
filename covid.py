#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import threading

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import arima
from pmdarima.model_selection import train_test_split
from sklearn.metrics import r2_score


def adjust_date(s):
    t = s.split("/")
    return f"20{t[2]}-{int(t[0]):02d}-{int(t[1]):02d}"


def draw(name, data):
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

    plt.plot(data.index, data, label="实际值", color="blue")
    plt.plot(validating.index, validating, label="校验值", color="orange")
    plt.plot(forecasting.index, forecasting, label="预测值", color="red")
    # plt.fill_between(forecasting.index, pred_ci[:, 0], pred_ci[:, 1], color="black", alpha=.25)

    plt.legend()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.title(f"Daily Increasement Forecasting - {name} (R2 = {r2:.6f})")
    plt.savefig(os.path.join("figures", f"covid-{name}.png"), bbox_inches="tight")


if __name__ == "__main__":
    # 准备数据
    df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    ).drop(columns=["Lat", "Long"]).groupby("Country/Region").sum().transpose()
    df.index = pd.DatetimeIndex(df.index.map(adjust_date))

    # 多线程
    countries = [
        "China",
        "US",
        "Russia",
        "Japan",
        "India",
        "United Kingdom",
        "Italy",
        "France",
        "Germany",
        "Spain",
        "Brazil",
    ]
    threads = []

    for country in countries:
        t = threading.Thread(target=draw, args=(country, df[country].diff().dropna()))
        threads.append(t)

    for t in threads:
        t.start()

    with codecs.open("README.md", "w", 'utf-8') as f:
        f.write("# CCOVID 预测\n\n")
        for country in countries:
            f.write(f"### {country}\n")
            f.write(f"![img](figures/covid-{country}.png)\n\n")
