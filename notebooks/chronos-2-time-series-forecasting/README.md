# Time Series Forecasting with Chronos-2 and OpenVINO™

Time series foundation models produce accurate forecasts for previously unseen data zero-shot, without training a separate model for every dataset. [Chronos-2](https://huggingface.co/amazon/chronos-2) is a time series foundation model with 120M parameters from Amazon, which supports **univariate**, **multivariate** and **covariate-informed** forecasting within a single architecture. The model is based on a T5-style encoder that alternates time attention along each series with group attention across related series (targets and covariates) and predicts quantiles for the whole forecast horizon in a single forward pass. At the time of release, Chronos-2 achieved state-of-the-art zero-shot accuracy on [fev-bench](https://huggingface.co/spaces/autogluon/fev-leaderboard) and [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) benchmarks.

This tutorial shows how to convert Chronos-2 to OpenVINO™ IR, compress its weights with NNCF and use it as a drop-in replacement inside the original `Chronos2Pipeline` for hourly bike sharing demand forecasting.

More details about the model can be found in the [technical report](https://arxiv.org/abs/2510.15821), [model card](https://huggingface.co/amazon/chronos-2) and [GitHub repository](https://github.com/amazon-science/chronos-forecasting).

## Notebook Contents

The tutorial consists of the following steps:

- Install prerequisites and prepare the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (CC BY 4.0) with hourly rental counts, weather and calendar information.
- Run the original PyTorch model with `Chronos2Pipeline`.
- Convert the model to OpenVINO IR with dynamic shapes. Operations that are not supported by the OpenVINO PyTorch Frontend are replaced with mathematically equivalent ones during conversion.
- Compress model weights to INT8 using [NNCF](https://github.com/openvinotoolkit/nncf).
- Run univariate, covariate-informed (weather and calendar) and multivariate forecasting with OpenVINO.
- Compare accuracy and performance of PyTorch and OpenVINO models on a rolling backtest.
- Launch an interactive Gradio demo.

The model is distributed under the [Apache 2.0](https://huggingface.co/amazon/chronos-2) license.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/chronos-2-time-series-forecasting/README.md" />
