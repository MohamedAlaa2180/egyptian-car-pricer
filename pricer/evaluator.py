from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.notebook import tqdm

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

WORKERS = 5
DEFAULT_SIZE = 200

# Relative-error bands for EGP (course $40/$80 bands do not apply here).
GREEN_MAPE = 0.15
ORANGE_MAPE = 0.30


class Tester:
    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE, workers=WORKERS):
        self.predictor = predictor
        self.data = data
        self.title = title or self.make_title(predictor)
        self.size = min(size, len(data))
        self.titles = []
        self.guesses = []
        self.truths = []
        self.errors = []
        self.pct_errors = []
        self.colors = []
        self.workers = workers

    @staticmethod
    def make_title(predictor) -> str:
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    @staticmethod
    def post_process(value):
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "").replace("EGP", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", value)
            return float(match.group()) if match else 0.0
        return float(value)

    def color_for(self, error, truth):
        if truth <= 0:
            return "red"
        mape = error / truth
        if mape < GREEN_MAPE:
            return "green"
        if mape < ORANGE_MAPE:
            return "orange"
        return "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        value = self.predictor(datapoint)
        guess = self.post_process(value)
        truth = float(datapoint.price)
        error = abs(guess - truth)
        pct = error / truth if truth else 1.0
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."
        return title, guess, truth, error, pct, color

    def chart(self, title):
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "error": self.errors,
                "color": self.colors,
            }
        )
        df["hover"] = [
            f"{t}\nGuess=EGP {g:,.0f}\nActual=EGP {y:,.0f}"
            for t, g, y in zip(df["title"], df["guess"], df["truth"])
        ]
        max_val = float(max(df["truth"].max(), df["guess"].max()))

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Actual asking price (EGP)", "guess": "Predicted price (EGP)"},
            width=1000,
            height=800,
        )
        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6)

        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(width=2, dash="dash", color="deepskyblue"),
                name="y = x",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.update_xaxes(range=[0, max_val])
        fig.update_yaxes(range=[0, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def error_trend_chart(self):
        n = len(self.errors)
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n + 1))
        running_means = [s / i for s, i in zip(running_sums, x)]
        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]
        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [m - c for m, c in zip(running_means, ci)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="95% CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                name="Cumulative Avg Error",
                customdata=list(zip(ci)),
                hovertemplate=(
                    "n=%{x}<br>"
                    "Avg Error=EGP %{y:,.0f}<br>"
                    "±95% CI=EGP %{customdata[0]:,.0f}<extra></extra>"
                ),
            )
        )
        final_mean = running_means[-1]
        final_ci = ci[-1]
        mape = sum(self.pct_errors) / n * 100
        title = (
            f"{self.title} MAE: EGP {final_mean:,.0f} ± {final_ci:,.0f}  |  MAPE: {mape:.1f}%"
        )
        fig.update_layout(
            title=title,
            xaxis_title="Number of Datapoints",
            yaxis_title="Average Absolute Error (EGP)",
            width=1000,
            height=360,
            template="plotly_white",
            showlegend=False,
        )
        fig.show()

    def report(self):
        n = self.size
        mae = sum(self.errors) / n
        mape = sum(self.pct_errors) / n * 100
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100
        title = (
            f"{self.title} results<br>"
            f"<b>MAE:</b> EGP {mae:,.0f}  "
            f"<b>MAPE:</b> {mape:.1f}%  "
            f"<b>MSE:</b> {mse:,.0f}  "
            f"<b>r²:</b> {r2:.1f}%"
        )
        self.error_trend_chart()
        self.chart(title)

    def run(self):
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for title, guess, truth, error, pct, color in tqdm(
                ex.map(self.run_datapoint, range(self.size)), total=self.size
            ):
                self.titles.append(title)
                self.guesses.append(guess)
                self.truths.append(truth)
                self.errors.append(error)
                self.pct_errors.append(pct)
                self.colors.append(color)
                print(f"{COLOR_MAP[color]}EGP {error:,.0f} {RESET}", end="")
        print()
        self.report()


def evaluate(function, data, size=DEFAULT_SIZE, workers=WORKERS):
    Tester(function, data, size=size, workers=workers).run()
