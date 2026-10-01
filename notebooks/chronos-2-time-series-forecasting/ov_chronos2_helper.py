import gc
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import openvino as ov

MODEL_FILE_NAME = "openvino_model.xml"


# ---------------------------------------------------------------------------
# Export helpers
#
# Chronos-2 uses several PyTorch operations that are not supported by the
# OpenVINO PyTorch Frontend (`torch.nanmean`, `torch.nan_to_num`, `torch.arcsinh`,
# boolean `torch.einsum` and `Tensor.unfold` with dynamic shapes).
# The functions below are mathematically equivalent replacements built from
# supported operations. They are applied only for the time of model conversion.
# ---------------------------------------------------------------------------


def _nan_to_num(x: torch.Tensor, nan: float = 0.0, **kwargs) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.full_like(x, nan), x)


def _nanmean(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    valid = ~torch.isnan(x)
    total = torch.where(valid, x, torch.zeros_like(x)).sum(dim=dim, keepdim=keepdim)
    count = valid.to(x.dtype).sum(dim=dim, keepdim=keepdim)
    return total / count


def _arcsinh(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.abs(x)
    return torch.sign(x) * torch.log(abs_x + torch.sqrt(abs_x * abs_x + 1.0))


def _instance_norm_forward(self, x: torch.Tensor, loc_scale=None):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if loc_scale is None:
        loc = _nan_to_num(_nanmean(x, dim=-1, keepdim=True), nan=0.0)
        scale = _nan_to_num(_nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale == 0, torch.full_like(scale, self.eps), scale)
    else:
        loc, scale = loc_scale

    scaled_x = (x - loc) / scale

    if self.use_arcsinh:
        scaled_x = _arcsinh(scaled_x)

    return scaled_x.to(orig_dtype), (loc, scale)


def _patch_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Equivalent of left padding + unfold for non-overlapping patches (patch_stride == patch_size).
    # Zero padding is used instead of NaN: padded positions are masked out by the model anyway,
    # and NaN pad values are rejected by some OpenVINO device plugins.
    length = x.shape[-1]
    pad = (self.patch_size - length % self.patch_size) % self.patch_size
    x = F.pad(x, (pad, 0), value=0.0)
    return x.reshape(x.shape[0], -1, self.patch_size)


def _construct_and_invert_group_time_mask(group_ids: torch.Tensor, attention_mask: torch.Tensor, floating_type: torch.dtype) -> torch.Tensor:
    group_mask = (group_ids[:, None] == group_ids[None, :]).to(attention_mask.dtype)
    # outer product of group mask (q, b) and time mask (b, t) -> (q, b, t), then reshape to (t, 1, q, b)
    group_time_mask = group_mask[:, :, None] * attention_mask[None, :, :]
    group_time_mask = group_time_mask.permute(2, 0, 1).unsqueeze(1)
    return (1.0 - group_time_mask) * torch.finfo(floating_type).min


@contextmanager
def patch_chronos2_for_export():
    """
    Temporarily replaces Chronos-2 operations that are not supported by OpenVINO PyTorch Frontend
    with mathematically equivalent implementations.
    """
    from chronos import chronos_bolt
    from chronos.chronos2 import model as chronos2_model

    orig_instance_norm = chronos_bolt.InstanceNorm.forward
    orig_patch = chronos_bolt.Patch.forward
    orig_group_mask = chronos2_model.Chronos2Encoder.__dict__["_construct_and_invert_group_time_mask"]
    orig_nan_to_num = torch.nan_to_num

    chronos_bolt.InstanceNorm.forward = _instance_norm_forward
    chronos_bolt.Patch.forward = _patch_forward
    chronos2_model.Chronos2Encoder._construct_and_invert_group_time_mask = staticmethod(_construct_and_invert_group_time_mask)
    torch.nan_to_num = _nan_to_num
    try:
        yield
    finally:
        chronos_bolt.InstanceNorm.forward = orig_instance_norm
        chronos_bolt.Patch.forward = orig_patch
        chronos2_model.Chronos2Encoder._construct_and_invert_group_time_mask = orig_group_mask
        torch.nan_to_num = orig_nan_to_num


class Chronos2ForecastWrapper(torch.nn.Module):
    """
    Wraps `Chronos2Model` into a module with tensor-only inputs and outputs suitable for conversion.
    The number of predicted patches is derived from the length of `future_covariates`.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.output_patch_size = model.chronos_config.output_patch_size

    def forward(self, context: torch.Tensor, group_ids: torch.Tensor, future_covariates: torch.Tensor) -> torch.Tensor:
        num_output_patches = future_covariates.shape[-1] // self.output_patch_size
        output = self.model(
            context=context,
            group_ids=group_ids,
            future_covariates=future_covariates,
            num_output_patches=num_output_patches,
        )
        return output.quantile_preds


def convert_chronos2(model_id: str, output_dir: Path) -> Path:
    """
    Converts Chronos-2 PyTorch model to OpenVINO IR format.

    :param model_id: Hugging Face Hub model id or local path to Chronos-2 model
    :param output_dir: directory for saving OpenVINO model and model configuration
    :returns: path to OpenVINO model xml file
    """
    from chronos import Chronos2Pipeline

    output_dir = Path(output_dir)
    ov_model_path = output_dir / MODEL_FILE_NAME
    if ov_model_path.exists():
        print(f"OpenVINO model already exists in {output_dir}")
        return ov_model_path

    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    model = pipeline.model.eval()
    output_patch_size = model.chronos_config.output_patch_size

    batch_size = 3
    example_input = {
        "context": torch.randn(batch_size, 512),
        "group_ids": torch.tensor([0, 0, 1]),
        "future_covariates": torch.full((batch_size, 4 * output_patch_size), float("nan")),
    }
    # first two series form a group: the second one has known future values (covariate), the first one is a target
    example_input["future_covariates"][1] = torch.randn(4 * output_patch_size)

    # input validation checks in the original code produce expected tracer warnings, which are safe to ignore
    with torch.no_grad(), patch_chronos2_for_export(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ov_model = ov.convert_model(
            Chronos2ForecastWrapper(model),
            example_input=example_input,
            input=[
                ("context", ov.PartialShape([-1, -1]), ov.Type.f32),
                ("group_ids", ov.PartialShape([-1]), ov.Type.i64),
                ("future_covariates", ov.PartialShape([-1, -1]), ov.Type.f32),
            ],
        )
    ov_model.output(0).get_tensor().set_names({"quantile_preds"})
    ov.save_model(ov_model, ov_model_path)
    model.config.save_pretrained(output_dir)
    print(f"OpenVINO model saved to {output_dir}")

    del ov_model, pipeline, model
    gc.collect()
    return ov_model_path


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


class OVChronos2Model:
    """
    Drop-in replacement of `Chronos2Model` for `Chronos2Pipeline`, which runs inference with OpenVINO.
    """

    def __init__(self, model_dir: Path, device: str = "CPU", ov_config: dict = None, core: ov.Core = None):
        from chronos.chronos2.config import Chronos2CoreConfig, Chronos2ForecastingConfig

        model_dir = Path(model_dir)
        self.config = Chronos2CoreConfig.from_pretrained(model_dir)
        chronos_config = dict(self.config.chronos_config)
        chronos_config.setdefault("time_encoding_scale", chronos_config["context_length"])
        self.chronos_config = Chronos2ForecastingConfig(**chronos_config)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        if ov_config is None:
            ov_config = {}
            # Chronos-2 encoder (like the original T5) overflows in FP16, so execution in FP32 is enforced for GPU.
            if "GPU" in device or "AUTO" in device:
                ov_config["INFERENCE_PRECISION_HINT"] = "f32"
        core = core or ov.Core()
        self.compiled_model = core.compile_model(model_dir / MODEL_FILE_NAME, device, ov_config)
        self.request = self.compiled_model.create_infer_request()

    def eval(self):
        return self

    def __call__(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor = None,
        future_covariates: torch.Tensor = None,
        num_output_patches: int = 1,
        **kwargs,
    ):
        from chronos.chronos2.model import Chronos2Output

        batch_size = context.shape[0]
        context = context[..., -self.chronos_config.context_length :]
        output_size = num_output_patches * self.chronos_config.output_patch_size

        if group_ids is None:
            group_ids = torch.arange(batch_size)
        if future_covariates is None:
            future_covariates = torch.full((batch_size, output_size), float("nan"))
        elif future_covariates.shape[-1] < output_size:
            padding = torch.full((batch_size, output_size - future_covariates.shape[-1]), float("nan"))
            future_covariates = torch.cat([future_covariates, padding], dim=-1)
        else:
            future_covariates = future_covariates[..., :output_size]

        self.request.infer(
            {
                "context": np.ascontiguousarray(context.numpy(), dtype=np.float32),
                "group_ids": np.ascontiguousarray(group_ids.numpy(), dtype=np.int64),
                "future_covariates": np.ascontiguousarray(future_covariates.numpy(), dtype=np.float32),
            }
        )
        quantile_preds = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        return Chronos2Output(quantile_preds=quantile_preds)


def plot_forecast(
    context_df,
    forecast_df,
    target: str,
    ground_truth_df=None,
    timestamp_column: str = "timestamp",
    history_length: int = 24 * 7,
    title: str = "",
    ax=None,
):
    """
    Plots the tail of the historical series, the median forecast and 10%-90% prediction interval.

    :param context_df: data frame with historical values
    :param forecast_df: data frame returned by `Chronos2Pipeline.predict_df` with "0.1", "0.5" and "0.9" columns
    :param target: name of the forecasted column
    :param ground_truth_df: optional data frame with actual future values
    :param timestamp_column: name of the timestamp column
    :param history_length: number of last historical points to show
    :param title: plot title
    :param ax: optional matplotlib axes
    :returns: matplotlib figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3.5))
    else:
        fig = ax.figure
    history = context_df.tail(history_length)
    forecast = forecast_df[forecast_df["target_name"] == target]
    ax.plot(history[timestamp_column], history[target], color="tab:blue", label="history")
    if ground_truth_df is not None:
        ax.plot(ground_truth_df[timestamp_column], ground_truth_df[target], color="tab:green", label="ground truth")
    ax.plot(forecast[timestamp_column], forecast["0.5"], color="tab:orange", label="median forecast")
    ax.fill_between(forecast[timestamp_column], forecast["0.1"], forecast["0.9"], color="tab:orange", alpha=0.3, label="80% prediction interval")
    ax.set_title(title or target)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def load_ov_chronos2_pipeline(model_dir: Path, device: str = "CPU", ov_config: dict = None):
    """
    Creates `Chronos2Pipeline` which uses OpenVINO model for inference.

    :param model_dir: directory with converted OpenVINO model and model configuration
    :param device: OpenVINO inference device
    :param ov_config: optional OpenVINO compilation config
    :returns: Chronos2Pipeline instance
    """
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline(model=OVChronos2Model(model_dir, device=device, ov_config=ov_config))
