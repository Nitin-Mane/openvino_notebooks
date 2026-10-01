import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ov_chronos2_helper import plot_forecast


def make_demo(pipeline, data: pd.DataFrame, target: str, covariates: list, id_column: str = "item_id", timestamp_column: str = "timestamp"):
    max_days_back = len(data) // 24 - 60

    def forecast(days_back: int, context_days: int, prediction_length: int, use_covariates: bool):
        cutoff = len(data) - int(days_back) * 24
        context_length = int(context_days) * 24
        prediction_length = int(prediction_length)
        context_df = data.iloc[max(0, cutoff - context_length) : cutoff]
        future_df = data.iloc[cutoff : cutoff + prediction_length]

        columns = [id_column, timestamp_column, target] + (covariates if use_covariates else [])
        forecast_df = pipeline.predict_df(
            context_df[columns],
            future_df=future_df[[id_column, timestamp_column] + covariates] if use_covariates else None,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
        )
        mae = np.abs(forecast_df["0.5"].to_numpy()[: len(future_df)] - future_df[target].to_numpy()).mean()
        title = f"Forecast from {future_df[timestamp_column].iloc[0]:%Y-%m-%d %H:%M} ({'with' if use_covariates else 'without'} covariates), MAE: {mae:.1f}"
        fig = plot_forecast(context_df, forecast_df, target=target, ground_truth_df=future_df, timestamp_column=timestamp_column, title=title)
        plt.close(fig)
        return fig

    demo = gr.Interface(
        fn=forecast,
        inputs=[
            gr.Slider(minimum=7, maximum=max_days_back, value=14, step=1, label="Forecast start (days before the end of the dataset)"),
            gr.Slider(minimum=7, maximum=341, value=85, step=1, label="Context length (days)"),
            gr.Slider(minimum=16, maximum=168, value=48, step=8, label="Prediction length (hours)"),
            gr.Checkbox(value=True, label="Use weather and calendar covariates"),
        ],
        outputs=gr.Plot(label="Forecast"),
        title="Bike sharing demand forecasting with Chronos-2 and OpenVINO",
        description="Select forecast start, history length and horizon. The orange band shows the 80% prediction interval.",
        flagging_mode="never",
    )
    return demo
