import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 이용해 선형회귀를 만들고, "
    "과거 데이터를 학습한 모델이 최근 기온을 얼마나 잘 예측하는지 비교합니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# =========================================================
# 연도별 평균기온 계산
# =========================================================
yearly = (
    df.dropna(subset=["연도", "평균기온"])
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년까지 + 관측일수 300일 이상
yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# =========================================================
# 전체 데이터 회귀
# =========================================================
# 독립변수: 1908년부터 지난 연수
yearly["경과연수"] = yearly["연도"] - 1908

X_all = yearly[["경과연수"]]
y_all = yearly["연평균기온"]

model_all = LinearRegression()
model_all.fit(X_all, y_all)

yearly["전체회귀예측"] = model_all.predict(X_all)

all_slope = model_all.coef_[0]
all_intercept = model_all.intercept_

all_r2 = r2_score(y_all, yearly["전체회귀예측"])


# =========================================================
# 50년 / 100년 학습 + 20년 테스트
# =========================================================

# 공통 테스트 데이터
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


def train_and_evaluate(start_year, end_year):
    train = yearly[
        (yearly["연도"] >= start_year) &
        (yearly["연도"] <= end_year)
    ].copy()

    X_train = train[["경과연수"]]
    y_train = train["연평균기온"]

    X_test = test[["경과연수"]]
    y_test = test["연평균기온"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "train": train,
        "predictions": predictions,
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "mae": mae,
        "mse": mse,
        "r2": r2
    }


model_50 = train_and_evaluate(1956, 2005)
model_100 = train_and_evaluate(1906, 2005)


# =========================================================
# 제목 및 데이터 정보
# =========================================================
st.divider()
st.header("1. 전체 데이터로 본 서울 연평균기온")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "회귀에 사용한 연도 수",
        f"{len(yearly)}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{yearly['연도'].min()}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{yearly['연도'].max()}년"
    )

with col4:
    st.metric(
        "전체 데이터 R²",
        f"{all_r2:.3f}"
    )


st.write(
    f"관측일수가 300일 이상인 연도만 사용했으며, "
    f"2025년 이후 데이터는 제외했습니다. "
    f"회귀에 사용된 기간은 **{yearly['연도'].min()}~{yearly['연도'].max()}년**입니다."
)


# =========================================================
# 전체 데이터 산점도 + 회귀선
# =========================================================
fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)

# 1900~2100 회귀선
future_years = np.arange(1900, 2101)
future_x = future_years - 1908

future_all_pred = model_all.predict(
    pd.DataFrame({"경과연수": future_x})
)

fig_all.add_trace(
    go.Scatter(
        x=future_years,
        y=future_all_pred,
        mode="lines",
        name="전체 데이터 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600,
    hovermode="closest"
)

fig_all.update_xaxes(
    range=[1900, 2100],
    tickmode="linear",
    dtick=10,
    tickformat="d"
)

st.plotly_chart(fig_all, use_container_width=True)

st.write(
    f"전체 데이터 회귀선의 기울기는 **{all_slope:.4f}℃/년**입니다. "
    f"즉, 회귀선 기준으로 1년이 지날 때 연평균기온이 "
    f"약 **{all_slope:.4f}℃** 변하는 추세입니다."
)


# =========================================================
# 연도 슬라이더 예측
# =========================================================
st.divider()
st.header("2. 연도를 선택해서 예상 기온 확인하기")

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_x = selected_year - 1908

selected_prediction = model_all.predict(
    pd.DataFrame({"경과연수": [selected_x]})
)[0]

st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{selected_prediction:.2f} ℃"
)


# =========================================================
# 선택 연도 그래프
# =========================================================
fig_prediction = go.Figure()

fig_prediction.add_trace(
    go.Scatter(
        x=future_years,
        y=future_all_pred,
        mode="lines",
        name="회귀선",
        line=dict(width=3)
    )
)

fig_prediction.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[selected_prediction],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=15),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 연평균기온: {selected_prediction:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 실제 데이터가 존재하는 경우 실제값 표시
actual = yearly[yearly["연도"] == selected_year]

if not actual.empty:
    actual_temp = actual.iloc[0]["연평균기온"]

    fig_prediction.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[actual_temp],
            mode="markers",
            name=f"{selected_year}년 실제값",
            marker=dict(
                size=13,
                symbol="diamond"
            ),
            hovertemplate=(
                f"{selected_year}년<br>"
                f"실제 연평균기온: {actual_temp:.2f}℃"
                "<extra></extra>"
            )
        )
    )

fig_prediction.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=500
)

fig_prediction.update_xaxes(
    range=[1900, 2100],
    tickmode="linear",
    dtick=10,
    tickformat="d"
)

st.plotly_chart(
    fig_prediction,
    use_container_width=True
)


# =========================================================
# 학습 / 테스트 데이터 설명
# =========================================================
st.divider()
st.header("3. 과거 데이터로 학습하고 최근 20년 예측하기")

st.write(
    """
두 개의 선형회귀 모델을 만들었습니다.

- **최근 50년 모델:** 1956~2005년 데이터를 학습
- **최근 100년 모델:** 1906~2005년 데이터를 학습
- **공통 테스트 데이터:** 2006~2025년

두 모델을 똑같은 2006~2025년 데이터에 적용하여
어느 기간으로 학습한 회귀선이 최근 기온을 더 잘 예측하는지 비교합니다.
"""
)


# =========================================================
# 기울기 비교
# =========================================================
st.subheader("📐 회귀선 기울기 비교")

slope_table = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "훈련 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "기울기 (℃/년)": [
        model_50["slope"],
        model_100["slope"]
    ]
})

slope_table["기울기 (℃/년)"] = slope_table[
    "기울기 (℃/년)"
].round(4)

st.dataframe(
    slope_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 성능 평가
# =========================================================
st.subheader("🎯 최근 20년 예측 성능")

performance = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "훈련 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 기간": [
        "2006~2025",
        "2006~2025"
    ],
    "MAE": [
        model_50["mae"],
        model_100["mae"]
    ],
    "MSE": [
        model_50["mse"],
        model_100["mse"]
    ],
    "R²": [
        model_50["r2"],
        model_100["r2"]
    ]
})

performance["MAE"] = performance["MAE"].round(3)
performance["MSE"] = performance["MSE"].round(3)
performance["R²"] = performance["R²"].round(3)

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 가장 좋은 모델 판단
# =========================================================
mae_50 = model_50["mae"]
mae_100 = model_100["mae"]

mse_50 = model_50["mse"]
mse_100 = model_100["mse"]

r2_50 = model_50["r2"]
r2_100 = model_100["r2"]

score_50 = 0
score_100 = 0

if mae_50 < mae_100:
    score_50 += 1
else:
    score_100 += 1

if mse_50 < mse_100:
    score_50 += 1
else:
    score_100 += 1

if r2_50 > r2_100:
    score_50 += 1
else:
    score_100 += 1


st.subheader("🔎 결과 해석")

if score_50 > score_100:
    better = "최근 50년 학습 모델"
elif score_100 > score_50:
    better = "최근 100년 학습 모델"
else:
    better = "두 모델이 비슷한 성능"


st.write(
    f"세 가지 평가 지표를 종합하면 **{better}**이 "
    f"2006~2025년 기온을 더 잘 예측한 것으로 나타났습니다."
)

st.write(
    f"- 최근 50년 모델의 기울기: **{model_50['slope']:.4f}℃/년**"
)

st.write(
    f"- 최근 100년 모델의 기울기: **{model_100['slope']:.4f}℃/년**"
)

st.write(
    f"- 최근 50년 모델 MAE: **{mae_50:.3f}℃**"
)

st.write(
    f"- 최근 100년 모델 MAE: **{mae_100:.3f}℃**"
)


# =========================================================
# 50년 / 100년 회귀선 + 테스트 실제값
# =========================================================
st.subheader("📈 50년 학습과 100년 학습 회귀선 비교")

compare_years = np.arange(1900, 2101)
compare_x = compare_years - 1908

pred_50 = model_50["model"].predict(
    pd.DataFrame({"경과연수": compare_x})
)

pred_100 = model_100["model"].predict(
    pd.DataFrame({"경과연수": compare_x})
)

fig_compare = go.Figure()

# 훈련 데이터
fig_compare.add_trace(
    go.Scatter(
        x=model_50["train"]["연도"],
        y=model_50["train"]["연평균기온"],
        mode="markers",
        name="50년 학습 데이터",
        marker=dict(size=5)
    )
)

fig_compare.add_trace(
    go.Scatter(
        x=model_100["train"]["연도"],
        y=model_100["train"]["연평균기온"],
        mode="markers",
        name="100년 학습 데이터",
        marker=dict(size=4)
    )
)

# 테스트 실제값
fig_compare.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="markers",
        name="테스트 실제값 (2006~2025)",
        marker=dict(size=9)
    )
)

# 50년 회귀선
fig_compare.add_trace(
    go.Scatter(
        x=compare_years,
        y=pred_50,
        mode="lines",
        name="1956~2005 회귀선",
        line=dict(width=3)
    )
)

# 100년 회귀선
fig_compare.add_trace(
    go.Scatter(
        x=compare_years,
        y=pred_100,
        mode="lines",
        name="1906~2005 회귀선",
        line=dict(width=3, dash="dash")
    )
)

fig_compare.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=650,
    hovermode="closest"
)

fig_compare.update_xaxes(
    range=[1900, 2100],
    tickmode="linear",
    dtick=10,
    tickformat="d"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# =========================================================
# 테스트 데이터별 예측값
# =========================================================
st.subheader("📋 2006~2025년 실제값과 예측값")

result_table = test[
    ["연도", "연평균기온"]
].copy()

result_table["50년 모델 예측"] = model_50["predictions"]
result_table["100년 모델 예측"] = model_100["predictions"]

result_table["50년 모델 오차"] = (
    result_table["연평균기온"] -
    result_table["50년 모델 예측"]
)

result_table["100년 모델 오차"] = (
    result_table["연평균기온"] -
    result_table["100년 모델 예측"]
)

for column in result_table.columns[1:]:
    result_table[column] = result_table[column].round(2)

st.dataframe(
    result_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 설명
# =========================================================
st.divider()

st.caption(
    "※ 연평균기온은 일별 평균기온을 연도별로 평균하여 계산했습니다. "
    "관측일수가 300일 미만인 연도와 2025년 이후 데이터는 분석에서 제외했습니다."
)

st.caption(
    "※ MAE와 MSE는 낮을수록 좋고, R²는 높을수록 좋습니다. "
    "테스트 데이터는 두 모델 모두 2006~2025년으로 동일하게 사용했습니다."
)

st.caption(
    "※ 선형회귀의 예측값은 과거 기온의 선형적인 추세를 바탕으로 계산한 값이며, "
    "실제 미래 기온을 보장하지 않습니다."
)
