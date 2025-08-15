import json
import re
import os
import subprocess
from pathlib import Path
from typing import Optional, Literal

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------- utils ----------------------
def _expected_ckpt(model_dir: Path) -> Path:
    return model_dir / "checkpoints" / "last.ckpt"

def _is_lfs_pointer(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            head = f.read(200)
        return head.startswith(b"version ") and b"git-lfs.github.com" in head
    except Exception:
        return False

def _validate_ckpt_by_model_name(model_name: str) -> tuple[bool, str]:
    """
    返回 (ok, msg)。ok=False 时 msg 为明确的错误提示。
    """
    model_dir = (CHECKPOINT_ROOT / model_name).resolve()
    ckpt = _expected_ckpt(model_dir)
    if not ckpt.exists():
        return False, f"[错误] 未找到权重: {ckpt}\n请确认 {model_dir}/checkpoints/last.ckpt 是否存在。"
    try:
        size = ckpt.stat().st_size
    except Exception:
        size = 0
    if size < 1_000_000:
        return False, f"[错误] 权重文件过小，可能损坏: {ckpt} ({size} bytes)"
    if _is_lfs_pointer(ckpt):
        return False, (
            f"[错误] 检测到 Git LFS 指针文件: {ckpt}\n"
            f"请在仓库目录执行：\n"
            f"- Linux/WSL: git lfs install && git lfs pull\n"
            f"- Windows PowerShell: git lfs install; git lfs pull"
        )
    return True, str(ckpt)

def _norm_path(p: Optional[str]) -> str:
    if not p:
        pp = DEFAULT_RESULTS_DIR
    else:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (PROJECT_ROOT / pp).resolve()
    pp.mkdir(parents=True, exist_ok=True)
    return str(pp)

def scan_models(include_all: bool = True):
    """
    include_all=True 时，尽量列出潜在模型目录（包含 checkpoints/ 或 config.yaml 或存在 ckpt 文件）；
    否则只列出检测到有效 last.ckpt 的目录。
    """
    checkpoint_dir = CHECKPOINT_ROOT
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        models = []
        for d in checkpoint_dir.iterdir():
            if not d.is_dir():
                continue
            ckpt = _expected_ckpt(d)
            has_ckpt = ckpt.exists()
            valid_ckpt = has_ckpt and ckpt.stat().st_size > 1_000_000 and not _is_lfs_pointer(ckpt)
            if include_all:
                if (d / "checkpoints").exists() or (d / "config.yaml").exists() or has_ckpt:
                    models.append(d.name)
            else:
                if valid_ckpt:
                    models.append(d.name)
        return sorted(models)
    except Exception:
        return []

def get_model_dropdown_update():
    models = scan_models(include_all=True)
    return gr.update(choices=models, value=(models[0] if models else None))

def scan_result_files(results_path: str):
    """
    按参考代码方式：仅列出结果目录下的文件（不递归），按修改时间倒序，返回文件绝对路径列表，以便 gr.Files 显示文件名等信息。
    """
    try:
        rp = Path(_norm_path(results_path))
        files = [p for p in rp.iterdir() if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p) for p in files]
    except Exception:
        return []

def _list_to_cli_list(items: list[str]) -> str:
    # 变成 "['a','b']" 这种 Fire 友好的形式
    items = [s for s in (i.strip() for i in items) if s]
    inner = ",".join([f"'{s}'" for s in items])
    return f"[{inner}]"

def _json_no_spaces(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

# 新增：解析 chemical_system 文本为元素列表
def _parse_chem_sys(text: Optional[str]) -> Optional[list[str]]:
    """
    支持：
    - 'Ti-Al-Ni-Au' / 'Ti,Al,Ni,Au' / 'Ti Al Ni Au'
    - '["Ti","Al","Ni","Au"]'（JSON 列表）
    返回：['Ti','Al','Ni','Au']；无有效元素返回 None
    """
    if not text:
        return None
    s = text.strip()
    # 优先尝试 JSON 列表
    if s.startswith("[") and s.endswith("]"):
        try:
            val = json.loads(s)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                return [x[:1].upper() + x[1:].lower() for x in val if x.strip()]
        except Exception:
            pass
    # 按非字母分隔
    import re as _re
    tokens = [t for t in _re.split(r"[^A-Za-z]+", s) if t]
    elems = [t[:1].upper() + t[1:].lower() for t in tokens]
    return elems or None

def _flatten_chem_list(x) -> list[str]:
    """将任意嵌套的列表/元组扁平化为纯元素符号字符串列表，严格过滤非字符串元素"""
    out: list[str] = []
    stack = [x]  # 用栈实现深度优先遍历，处理任意层嵌套
    while stack:
        item = stack.pop()
        # 若为列表/元组，继续入栈解析内部元素
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))  # 保持原有顺序
        # 仅保留非空字符串，并标准化元素符号格式（首字母大写）
        elif isinstance(item, str):
            normalized = item.strip().capitalize()  # 统一格式为"Li"而非"li"或"LI"
            if normalized:  # 过滤空字符串
                out.append(normalized)
        # 忽略其他类型（如数字、字典等无效元素）
    # 去重并保持首次出现顺序
    seen = set()
    unique_elems = []
    for elem in out:
        if elem not in seen:
            seen.add(elem)
            unique_elems.append(elem)
    return unique_elems

def build_command(
    output_path: str,
    model_name: Optional[str],
    batch_size: int,
    num_batches: int,
    checkpoint_epoch: str,
    diffusion_guidance_factor: Optional[float],
    strict_checkpoint_loading: bool,
    record_trajectories: bool,
    properties_chem_sys: Optional[str],
    properties_eah: Optional[float],
    sampling_config_path: Optional[str],
    sampling_config_name: str,
    sampling_config_overrides_text: str,
    config_overrides_text: str,
    target_compositions_text: str,
) -> str:
    cmd = ["mattergen-generate", f'"{_norm_path(output_path)}"']

    if model_name:
        cmd.append(f'--model_path="{str((CHECKPOINT_ROOT / model_name).resolve())}"')

    cmd.append(f"--batch_size={batch_size}")
    cmd.append(f"--num_batches={num_batches}")

    # checkpoint
    if checkpoint_epoch:
        cmd.append(f"--checkpoint_epoch={checkpoint_epoch}")

    # properties_to_condition_on
    props = {}
    chem_list = _parse_chem_sys(properties_chem_sys)
    if chem_list:
        props["chemical_system"] = _flatten_chem_list(chem_list)
    if properties_eah is not None:
        props["energy_above_hull"] = float(properties_eah)
    if props:
        cmd.append(f"--properties_to_condition_on='{_json_no_spaces(props)}'")

    # guidance
    if diffusion_guidance_factor is not None:
        cmd.append(f"--diffusion_guidance_factor={diffusion_guidance_factor}")

    # strict loading / record trajectories
    if not strict_checkpoint_loading:
        cmd.append("--strict_checkpoint_loading=False")
    if not record_trajectories:
        cmd.append("--record_trajectories=False")

    # sampling config
    if sampling_config_path:
        cmd.append(f'--sampling_config_path="{sampling_config_path}"')
    if sampling_config_name:
        cmd.append(f"--sampling_config_name={sampling_config_name}")

    if sampling_config_overrides_text.strip():
        lines = [ln.strip() for ln in sampling_config_overrides_text.strip().splitlines() if ln.strip()]
        cmd.append(f'--sampling_config_overrides="{_list_to_cli_list(lines)}"')

    # model config overrides
    if config_overrides_text.strip():
        lines = [ln.strip() for ln in config_overrides_text.strip().splitlines() if ln.strip()]
        cmd.append(f'--config_overrides="{_list_to_cli_list(lines)}"')

    # target compositions (JSON)
    if target_compositions_text.strip():
        try:
            tc = json.loads(target_compositions_text)
            cmd.append(f"--target_compositions='{_json_no_spaces(tc)}'")
        except Exception:
            # 保留原文，交给 CLI 自己解析
            cmd.append(f"--target_compositions='{target_compositions_text.strip()}'")

    return " ".join(cmd)

def execute_command(cmd: str, results_path: str, model_name: Optional[str]):
    try:
        # 运行前校验 checkpoint（当使用 model_path 时）
        if model_name:
            ok, msg = _validate_ckpt_by_model_name(model_name)
            if not ok:
                return msg, scan_result_files(results_path)

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        out = f"命令执行成功！\n\n输出：\n{result.stdout}"
        if result.stderr.strip():
            out += f"\n\n[stderr]\n{result.stderr}"
        return out, scan_result_files(results_path)
    except subprocess.CalledProcessError as e:
        return f"执行出错：\n{e.stderr or e.stdout}", scan_result_files(results_path)

def run_via_api(
    output_path, model_name, batch_size, num_batches, checkpoint_epoch,
    diffusion_guidance_factor, strict_checkpoint_loading, record_trajectories,
    properties_chem_sys, properties_eah, sampling_config_path, sampling_config_name,
    sampling_config_overrides_text, config_overrides_text, target_compositions_text
):
    from mattergen.scripts.generate import main as generate_main
    import traceback

    # 新增：调用前先校验 checkpoint，避免 torch.load 报 invalid load key 'v'
    if model_name:
        ok, msg = _validate_ckpt_by_model_name(model_name)
        if not ok:
            return msg, scan_result_files(output_path)
    kwargs = dict(
        output_path=_norm_path(output_path),
        batch_size=int(batch_size),
        num_batches=int(num_batches),
        checkpoint_epoch=str(checkpoint_epoch),
        record_trajectories=bool(record_trajectories),
        strict_checkpoint_loading=bool(strict_checkpoint_loading),
        sampling_config_name=(sampling_config_name or "default").strip() or "default",
    )
    if model_name:
        kwargs["model_path"] = str((CHECKPOINT_ROOT / model_name).resolve())

    # 条件属性：强制扁平化为 list[str]
    props = {}
    chem_list = _parse_chem_sys(properties_chem_sys)
    if chem_list:
        chem_list = _flatten_chem_list(chem_list)
        chem_list = [elem for elem in chem_list if isinstance(elem, str)]
        props["chemical_system"] = chem_list
    if properties_eah is not None:
        props["energy_above_hull"] = float(properties_eah)
    if props:
        kwargs["properties_to_condition_on"] = props
    if diffusion_guidance_factor is not None:
        kwargs["diffusion_guidance_factor"] = float(diffusion_guidance_factor)
    if sampling_config_path:
        kwargs["sampling_config_path"] = sampling_config_path.strip()
    if sampling_config_overrides_text and sampling_config_overrides_text.strip():
        kwargs["sampling_config_overrides"] = [ln.strip() for ln in sampling_config_overrides_text.splitlines() if ln.strip()]
    if config_overrides_text and config_overrides_text.strip():
        kwargs["config_overrides"] = [ln.strip() for ln in config_overrides_text.splitlines() if ln.strip()]
    if target_compositions_text and target_compositions_text.strip():
        try:
            kwargs["target_compositions"] = json.loads(target_compositions_text)
        except Exception:
            pass

    # 调试输出：确认最终传入的 props 形状
    debug = f"[DEBUG] properties_to_condition_on={json.dumps(kwargs.get('properties_to_condition_on', {}), ensure_ascii=False)}\n"

    try:
        structures = generate_main(**kwargs)
        return debug + f"生成成功：{len(structures)} 个结构，已保存到 {kwargs['output_path']}", scan_result_files(output_path)
    except Exception as e:
        import traceback as tb
        return debug + "执行出错（API）：\n" + "".join(tb.format_exception(e)), scan_result_files(output_path)

# ---------------------- UI ----------------------
with gr.Blocks(title="MatterGen 生成器") as demo:
    gr.Markdown("# MatterGen 生成器")

    with gr.Row():
        model_name = gr.Dropdown(
            label="模型目录",
            choices=scan_models(include_all=True),
            value=(scan_models(include_all=True)[0] if scan_models(include_all=True) else None),
            interactive=True,
            allow_custom_value=True
        )

    with gr.Row():
        batch_size = gr.Number(label="Batch Size", value=16, precision=0)
        num_batches = gr.Number(label="Num Batches", value=1, precision=0)
        checkpoint_epoch = gr.Dropdown(label="Checkpoint Epoch", choices=["last", "best"], value="last")

    with gr.Accordion("条件与采样设置", open=False):
        with gr.Row():
            properties_chem_sys = gr.Textbox(label="chemical_system（例如 Ti-Al-Ni-Au）", placeholder="可留空")
            properties_eah = gr.Number(label="energy_above_hull（eV/atom）", value=None, precision=4)
            diffusion_guidance_factor = gr.Number(label="diffusion_guidance_factor", value=None, precision=3)

        with gr.Row():
            strict_checkpoint_loading = gr.Checkbox(label="strict_checkpoint_loading", value=True)
            record_trajectories = gr.Checkbox(label="record_trajectories", value=True)

        with gr.Row():
            sampling_config_name = gr.Textbox(label="sampling_config_name", value="default")
            sampling_config_path = gr.Textbox(label="sampling_config_path（可留空）", placeholder="/abs/path/to/sampling")

        sampling_config_overrides_text = gr.Textbox(
            label="sampling_config_overrides（每行一条）",
            placeholder="condition_loader_partial.batch_size=8\nsampler_partial.N=500",
            lines=3
        )
        config_overrides_text = gr.Textbox(
            label="config_overrides（每行一条）",
            placeholder="lightning_module.model.hidden_dim=256\ntrainer.precision=16",
            lines=3
        )
        target_compositions_text = gr.Textbox(
            label="target_compositions（JSON，CSP 模型才支持）",
            placeholder='[{"Si":1,"O":2},{"Al":2,"O":3}]',
            lines=2
        )

    with gr.Row():
        results_path = gr.Textbox(label="输出目录", value=str(DEFAULT_RESULTS_DIR))
        preview_btn = gr.Button("预览命令", variant="secondary")
        run_btn = gr.Button("开始生成", variant="primary")
        refresh_files_btn = gr.Button("刷新文件列表")

    generated_command = gr.Textbox(label="生成命令", interactive=False)
    output = gr.Textbox(label="执行结果", lines=12, interactive=False)
    file_list = gr.Files(label="生成结果文件")

    with gr.Accordion("参数帮助", open=False):
        gr.Markdown(
            "- 模型目录：checkpoints 下的子目录名。会优先使用 --model_path 指向该目录下 checkpoints/last.ckpt。\n"
            "- Batch Size：每个 batch 生成的样本数量。总样本数 ≈ batch_size × num_batches。\n"
            "- Num Batches：生成的 batch 数。与 batch_size 共同决定生成总量。\n"
            "- Checkpoint Epoch：选择 'last' 或 'best' 等 checkpoint 标签，传入 --checkpoint_epoch。\n"
            "- chemical_system：以元素集合限定搜索空间。格式可为：'Ti-Al-Ni'、'Ti,Al,Ni'、'Ti Al Ni'，或 JSON 列表 [\"Ti\",\"Al\",\"Ni\"]。\n"
            "- energy_above_hull（eV/atom）：目标上凸包能量约束。越小越稳定；作为条件传入。\n"
            "- diffusion_guidance_factor（γ）：条件引导强度。数值大更遵循条件（但可能牺牲多样性），数值小更发散（多样性更高）。\n"
            "- strict_checkpoint_loading：是否严格匹配权重与模型结构。不匹配时可关闭放宽载入。\n"
            "- record_trajectories：是否记录采样轨迹（用于分析/可视化）。关闭可减少 IO。\n"
            "- sampling_config_name：采样配置名（如 'default'）。用于内部选择一套默认采样参数。\n"
            "- sampling_config_path：采样配置文件路径。指定时优先生效，可覆盖默认配置。\n"
            "- sampling_config_overrides：逐行的 key=value 覆盖项，最终拼成可被 Fire 解析的列表传给 --sampling_config_overrides。\n"
            "- config_overrides：模型/训练/推理等更底层配置的覆盖项，格式同上，传给 --config_overrides。\n"
            "- target_compositions（JSON）：仅 CSP 类模型支持，指定目标化学计量（如 [{\"Si\":1,\"O\":2}]）。"
        )

    # 交互逻辑
    def on_preview(
        model_name, batch_size, num_batches, checkpoint_epoch,
        diffusion_guidance_factor, strict_checkpoint_loading, record_trajectories,
        properties_chem_sys, properties_eah, sampling_config_path, sampling_config_name,
        sampling_config_overrides_text, config_overrides_text, target_compositions_text,
        results_path
    ):
        cmd = build_command(
            output_path=results_path,
            model_name=model_name,
            batch_size=int(batch_size),
            num_batches=int(num_batches),
            checkpoint_epoch=str(checkpoint_epoch),
            diffusion_guidance_factor=diffusion_guidance_factor,
            strict_checkpoint_loading=bool(strict_checkpoint_loading),
            record_trajectories=bool(record_trajectories),
            properties_chem_sys=properties_chem_sys.strip() if properties_chem_sys else None,
            properties_eah=properties_eah if properties_eah is not None else None,
            sampling_config_path=sampling_config_path.strip() if sampling_config_path else None,
            sampling_config_name=sampling_config_name.strip() or "default",
            sampling_config_overrides_text=sampling_config_overrides_text,
            config_overrides_text=config_overrides_text,
            target_compositions_text=target_compositions_text,
        )
        return cmd

    preview_btn.click(
        fn=on_preview,
        inputs=[
            model_name, batch_size, num_batches, checkpoint_epoch,
            diffusion_guidance_factor, strict_checkpoint_loading, record_trajectories,
            properties_chem_sys, properties_eah, sampling_config_path, sampling_config_name,
            sampling_config_overrides_text, config_overrides_text, target_compositions_text,
            results_path
        ],
        outputs=[generated_command]
    )

    run_btn.click(
        fn=run_via_api,
        inputs=[
            results_path, model_name, batch_size, num_batches, checkpoint_epoch,
            diffusion_guidance_factor, strict_checkpoint_loading, record_trajectories,
            properties_chem_sys, properties_eah, sampling_config_path, sampling_config_name,
            sampling_config_overrides_text, config_overrides_text, target_compositions_text
        ],
        outputs=[output, file_list]
    )

    # 初次加载时刷新模型下拉
    demo.load(fn=get_model_dropdown_update, inputs=None, outputs=model_name)

    refresh_files_btn.click(fn=scan_result_files, inputs=[results_path], outputs=[file_list])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)