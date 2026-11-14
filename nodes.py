"""图像信息 —— 两个节点：读取图像信息（可直连分辨率选择器）、图像规范。

安装：把整个目录放进 ComfyUI/custom_nodes/，重启 ComfyUI。
"""

from __future__ import annotations

import logging
import math
from math import gcd

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 与原生 ResolutionSelector 对接
#
# ResolutionSelector 的 aspect_ratio 是 8 项下拉，选项字符串逐字校验（连错会报
# "Value not in list"），所以本节点的「宽高比」输出必须原样吐这些字符串。
# ---------------------------------------------------------------------------
RESOLUTION_OPTIONS = (
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
)

_OPTION_RATIOS = (
    (1, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (9, 16),
    (16, 9),
    (21, 9),
)

# ResolutionSelector 的 megapixels 输入范围 0.1 ~ 16.0，超出会被后端判超限
MIN_MEGAPIXELS = 0.1
MAX_MEGAPIXELS = 16.0

def _image_hw(image):
    """返回 (宽, 高, 批次)，兼容 [B, H, W, C] 张量。"""
    if image is None or not hasattr(image, "shape") or image.dim() < 3:
        raise ValueError("图像信息：需要一个有效的「图像」输入（[B, H, W, C] 张量）")
    shape = image.shape
    batch = int(shape[0]) if shape[0] else 1
    height = int(shape[-3])
    width = int(shape[-2])
    return width, height, batch


def _encoder_resolution(width: int, height: int, multiple: int = 32) -> int:
    """「文本编码 Qwen 图像 2.1」的分辨率输入值。

    那个节点会把每张参考图缩放到约「分辨率 × 分辨率」像素（边长按 32 对齐、保持比例）。
    所以这里给的是面积等效边长 √(宽×高)，不是长边、也不是像素总数；
    并在其附近微调，让编码器算出的参考图尺寸尽量等于规范化后的宽高。

    对齐粒度跟随「对齐倍数」输入：倍数不是 32 的整数倍时，算出来的是「按该倍数对齐
    时的等效值」，而 Qwen 编码器本身固定按 32 对齐（comfy_extras/nodes_qwen.py:157-158），
    所以实际参考图仍会按 32 取整，只能是近似。
    """
    width = max(1, int(width))
    height = max(1, int(height))
    alignment = max(1, min(256, int(multiple)))
    ideal = math.sqrt(width * height)
    ratio = width / height

    def encoder_size(res):
        # 与 comfy_extras/nodes_qwen.py:157-158 同一套算法，只是把固定的 32 换成对齐倍数
        w = round(math.sqrt(res * res * ratio) / alignment) * alignment
        h = round(math.sqrt(res * res / ratio) / alignment) * alignment
        return max(alignment, w), max(alignment, h)

    best_res, best_cost = max(alignment, min(4096, int(round(ideal)))), None
    for delta in range(-64, 65, 4):
        res = int(round(ideal)) + delta
        if res < alignment or res > 4096:
            continue
        w, h = encoder_size(res)
        cost = (abs(w - width) + abs(h - height), abs(res - ideal))
        if best_cost is None or cost < best_cost:
            best_res, best_cost = res, cost
    return best_res


def _encoder_note(multiple: int) -> str:
    """参考分辨率后面的提示：倍数不是 32 的整数倍时，编码器实际仍按 32 对齐。"""
    step = max(1, int(multiple))
    return "" if step % 32 == 0 else f"（按 {step} 对齐估算；Qwen 编码器固定 32 对齐）"


def _nearest_option(ratio: float) -> str:
    """在 ResolutionSelector 的 8 个画幅里取最接近的一个（对数距离，避免极端比例拉偏）。"""
    if ratio <= 0:
        return RESOLUTION_OPTIONS[0]
    best_index, best_dist = 0, float("inf")
    for index, (pw, ph) in enumerate(_OPTION_RATIOS):
        dist = abs(math.log(ratio / (pw / ph)))
        if dist < best_dist:
            best_index, best_dist = index, dist
    return RESOLUTION_OPTIONS[best_index]


class ImageAspectRatio(io.ComfyNode):
    """读取图像信息，输出可直接接「分辨率选择器」。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageAspectRatio",
            display_name="图像信息",
            category="图像/分析",
            description=(
                "读取图像的宽度、高度、宽高比、百万像素和尺寸文本。"
                "「宽高比」输出的是分辨率选择器下拉里的对应选项（取最接近的画幅），"
                "「百万像素」也已限制在分辨率选择器的允许范围内，两个输出都能直接连过去。"
            ),
            search_aliases=["aspect ratio", "图像信息", "图像比例", "宽高比", "ratio", "图像尺寸"],
            is_output_node=True,
            inputs=[
                io.Image.Input("image", display_name="图像"),
            ],
            outputs=[
                io.Int.Output("width", display_name="宽度", tooltip="图像宽度（像素）"),
                io.Int.Output("height", display_name="高度", tooltip="图像高度（像素）"),
                io.Combo.Output(
                    "aspect_ratio",
                    display_name="宽高比",
                    tooltip="最接近的标准画幅；字符串与「分辨率选择器」的 16:9 (Widescreen) 等选项完全一致，可直接连接",
                ),
                io.Float.Output(
                    "megapixels",
                    display_name="百万像素",
                    tooltip="宽 × 高 ÷ 1024 ÷ 1024；已限制在 0.1 ~ 16.0，可直接连接「分辨率选择器」",
                ),
                io.String.Output(
                    "resolution",
                    display_name="分辨率",
                    tooltip="宽高拼成的一行文本，如 1920x1080；可接文件名、文本拼接、预览类节点",
                ),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        width, height, _batch = _image_hw(image)
        if width <= 0 or height <= 0:
            raise ValueError(f"图像信息：图像尺寸非法 {width}×{height}")

        ratio = width / height
        divisor = gcd(width, height) or 1
        ratio_text = f"{width // divisor}:{height // divisor}"
        option = _nearest_option(ratio)

        megapixels = width * height / (1024 * 1024)
        clamped = min(max(megapixels, MIN_MEGAPIXELS), MAX_MEGAPIXELS)
        note = ""
        if abs(clamped - megapixels) > 1e-9:
            note = f"（百万像素已限制为 {clamped:.2f}，分辨率选择器上限 {MAX_MEGAPIXELS:g}）"

        text = f"{width}×{height} · {ratio_text} · {megapixels:.2f} 百万像素 → 分辨率选择器：{option}{note}"
        return io.NodeOutput(
            width,
            height,
            option,
            round(clamped, 4),
            f"{width}x{height}",
            ui={"text": (text,)},
        )


NORMALIZE_STRATEGIES = ["最小像素改动", "最小比例偏差", "只放大", "只缩小"]


def _normalize_size(width, height, multiple, strategy):
    """按对齐倍数把宽高规范化；策略取自 NORMALIZE_STRATEGIES。"""

    def sides(value):
        down = max(multiple, (value // multiple) * multiple)
        up = max(multiple, int(math.ceil(value / multiple)) * multiple)
        return down, up

    if strategy == "只放大":
        return int(math.ceil(width / multiple)) * multiple, int(math.ceil(height / multiple)) * multiple
    if strategy == "只缩小":
        return max(multiple, (width // multiple) * multiple), max(multiple, (height // multiple) * multiple)

    combos = {(w, h) for w in sides(width) for h in sides(height)}
    ideal = width / height
    prefer_ratio = strategy == "最小比例偏差"

    def score(size):
        w, h = size
        ratio_error = abs(w / h - ideal) / ideal
        pixel_change = abs(w - width) + abs(h - height)
        area_error = abs(w * h - width * height)
        if prefer_ratio:
            return (round(ratio_error, 6), pixel_change, area_error)
        return (pixel_change, round(ratio_error, 6), area_error)

    return min(combos, key=score)


class NormalizeImage(io.ComfyNode):
    """把图像重采样（或裁切）成规范化尺寸，直接输出可用的图片。"""

    METHODS = ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]
    MODES = ["缩放", "裁切"]

    @classmethod
    def define_schema(cls):
        return cls._schema()

    @classmethod
    def _schema(cls):
        return io.Schema(
            node_id="NormalizeImage",
            display_name="图像规范",
            category="图像/分析",
            description=(
                "把图像改成对齐倍数（Qwen-Image / Flux 为 16，SDXL 为 8）的整数倍，"
                "并直接输出改好的图片。处理方式可选「缩放」"
                "（重采样，构图比例基本不变）或「裁切」（居中裁掉多出的几个像素，不重采样）。"
            ),
            search_aliases=["normalize image", "图像规范", "规范化图像", "图像对齐", "缩放尺寸", "resize"],
            is_output_node=True,
            inputs=[
                io.Image.Input("image", display_name="图像"),
                io.Int.Input(
                    "multiple",
                    default=16,
                    min=1,
                    max=256,
                    display_name="对齐倍数",
                    tooltip="Qwen-Image / Flux 用 16，SD1.5 / SDXL 用 8",
                ),
                io.Combo.Input(
                    "strategy",
                    options=NORMALIZE_STRATEGIES,
                    default="最小像素改动",
                    display_name="策略",
                    tooltip="最小像素改动 / 最小比例偏差 / 只放大 / 只缩小",
                ),
                io.Combo.Input(
                    "mode",
                    options=cls.MODES,
                    default="缩放",
                    display_name="处理方式",
                    tooltip="缩放 = 重采样到目标尺寸；裁切 = 居中裁掉多出的像素（只能缩小，放大时会自动退回缩放）",
                ),
                io.Combo.Input(
                    "method",
                    options=cls.METHODS,
                    default="lanczos",
                    display_name="缩放算法",
                    tooltip="仅「缩放」时生效；lanczos 质量最好",
                ),
            ],
            outputs=[
                io.Image.Output("image", display_name="图像", tooltip="尺寸已规范的图像"),
                io.Int.Output("width", display_name="宽度", tooltip="输出图像宽度"),
                io.Int.Output("height", display_name="高度", tooltip="输出图像高度"),
                io.Int.Output(
                    "encoder_resolution",
                    display_name="参考分辨率",
                    tooltip="√(宽×高) 取整（像素）；对齐粒度跟随「对齐倍数」；可直接连「文本编码 Qwen 图像 2.1」的「分辨率」输入，范围 0~4096",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, multiple, strategy, mode="缩放", method="lanczos") -> io.NodeOutput:
        src_w, src_h, _batch = _image_hw(image)
        step = max(1, int(multiple))
        if strategy not in NORMALIZE_STRATEGIES:
            strategy = NORMALIZE_STRATEGIES[0]

        new_w, new_h = _normalize_size(src_w, src_h, step, strategy)
        deviation = ((new_w / new_h) - (src_w / src_h)) / (src_w / src_h) * 100.0

        note = ""
        if new_w == src_w and new_h == src_h:
            out = image
            note = "，尺寸已规范，未改动"
        else:
            out = None
            if mode == "裁切":
                out = _center_crop(image, new_w, new_h)
                if out is None:
                    note = "，目标尺寸比原图大，已退回缩放"
                else:
                    note = "，居中裁切未重采样"
            if out is None:
                out = _resize_image(image, new_w, new_h, method)
                note = note or f"，缩放 {method}"

        encoder_res = _encoder_resolution(new_w, new_h, step)
        text = (
            f"{src_w}×{src_h} → {new_w}×{new_h}（比例偏差 {deviation:+.2f}%） · {step} 对齐 · {strategy}"
            f"{note} · 参考分辨率 {encoder_res}{_encoder_note(step)}"
        )
        return io.NodeOutput(out, new_w, new_h, encoder_res, ui={"text": (text,)})


def _resize_image(image, width, height, method="lanczos"):
    """与原生「缩放图像」同一套实现：优先 comfy.utils.common_upscale，失败再退回 torch。"""
    try:
        from comfy.utils import common_upscale

        return common_upscale(image.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)
    except Exception as error:  # 独立环境下没有 comfy.utils 时兜底
        LOG.warning("图像规范：common_upscale 不可用（%s），退回 torch 插值", error)
        import torch.nn.functional as F

        x = image.movedim(-1, 1)
        torch_method = method if method in ("bilinear", "bicubic", "area", "nearest") else "bilinear"
        kwargs = {} if torch_method in ("area", "nearest") else {"align_corners": False}
        if torch_method == "bicubic":
            kwargs["antialias"] = True
        return F.interpolate(x, size=(height, width), mode=torch_method, **kwargs).movedim(1, -1)


def _center_crop(image, width, height):
    """居中裁到目标尺寸；目标比原图大时返回 None。"""
    src_h, src_w = int(image.shape[-3]), int(image.shape[-2])
    if width > src_w or height > src_h:
        return None
    top = (src_h - height) // 2
    left = (src_w - width) // 2
    return image[..., top:top + height, left:left + width, :]


class ImageAspectRatioExtension(ComfyExtension):
    @override
    async def get_node_list(cls) -> list[type[io.ComfyNode]]:
        return [ImageAspectRatio, NormalizeImage]


async def comfy_entrypoint() -> ImageAspectRatioExtension:
    return ImageAspectRatioExtension()