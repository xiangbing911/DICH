import argparse
import os
import re
import sys
from datetime import datetime
from typing import Optional, Tuple


def _safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text).strip("_")


def _resolve_film_type_args(
    *, film_type_cell: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (film_type_key, film_type_custom).
    - If matches a preset key or preset label, use --film-type.
    - Otherwise use --film-type-custom with original cell.
    """
    import qwen_feiyi_to_csv as gen

    raw = (film_type_cell or "").strip()
    if not raw:
        return "documentary", None

    if raw in gen.FILM_TYPE_PRESETS:
        return raw, None

    for k, label in gen.FILM_TYPE_PRESETS.items():
        if raw == label:
            return k, None

    return None, raw


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Read 非遗项目汇总.xlsx (项目名称/短片类型/背景资料) and call qwen_feiyi_to_csv.py per row."
    )
    p.add_argument("--xlsx", default="非遗项目汇总.xlsx", help="输入 XLSX 路径")
    p.add_argument(
        "--xlsx-out",
        default=None,
        help="输出 XLSX 路径（可选；默认覆盖输入文件）",
    )
    p.add_argument("--sheet", default=None, help="Sheet 名称（默认第一个）")
    p.add_argument(
        "--start-row",
        type=int,
        default=2,
        help="数据起始行（默认 2，跳过表头）",
    )
    p.add_argument(
        "--write-title-col",
        type=int,
        default=4,
        help="将生成的故事标题写入第几列（默认 4）",
    )
    p.add_argument(
        "--write-title-header",
        default="生成故事标题",
        help="表头名称（默认：生成故事标题）",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建备份文件（默认会在同目录创建 .bak_<timestamp>.xlsx）",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join("outputs", "batch"),
        help="输出目录（默认 outputs/batch）",
    )
    p.add_argument("--model", default="qwen3-max", help="Qwen model name")
    p.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="DashScope OpenAI-compatible base URL",
    )
    p.add_argument("--api-key", default=None, help="API Key（可选，默认读 DASHSCOPE_API_KEY）")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    p.add_argument(
        "--scene-count",
        type=int,
        default=None,
        help="场景数量（可选；不填则用 qwen_feiyi_to_csv.py 默认值）",
    )
    args = p.parse_args(argv)

    try:
        import openpyxl  # type: ignore
    except Exception as e:
        print(
            "Missing dependency: openpyxl. Install via: pip install openpyxl\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 2

    import qwen_feiyi_to_csv as gen

    if not os.path.exists(args.xlsx):
        print(f"XLSX not found: {args.xlsx}", file=sys.stderr)
        return 2

    wb = openpyxl.load_workbook(args.xlsx, read_only=False, data_only=True)
    ws = wb[args.sheet] if args.sheet else wb.worksheets[0]

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.start_row >= 2 and args.write_title_col >= 1:
        header_cell = ws.cell(row=1, column=args.write_title_col)
        if not str(header_cell.value or "").strip():
            header_cell.value = args.write_title_header

    processed = 0
    failed = 0
    for row_idx in range(args.start_row, ws.max_row + 1):
        name = ws.cell(row=row_idx, column=1).value
        film_type_cell = ws.cell(row=row_idx, column=2).value
        background = ws.cell(row=row_idx, column=3).value

        topic = str(name or "").strip()
        if not topic:
            continue

        film_type_key, film_type_custom = _resolve_film_type_args(
            film_type_cell=str(film_type_cell or "")
        )
        background_text = str(background or "").strip() or None

        safe_topic = _safe_name(topic)
        out_path = os.path.join(args.out_dir, f"{row_idx:04d}_{safe_topic}_{ts}.csv")

        gen_args: list[str] = [
            "--topic",
            topic,
            "--out",
            out_path,
            "--model",
            args.model,
            "--base-url",
            args.base_url,
        ]
        if args.api_key:
            gen_args += ["--api-key", args.api_key]
        if args.no_stream:
            gen_args += ["--no-stream"]
        if args.scene_count is not None:
            gen_args += ["--scene-count", str(int(args.scene_count))]
        if film_type_key:
            gen_args += ["--film-type", film_type_key]
        if film_type_custom:
            gen_args += ["--film-type-custom", film_type_custom]
        if background_text:
            gen_args += ["--background", background_text]

        print(f"\n=== Row {row_idx}: {topic} / {film_type_custom or film_type_key or ''} ===")
        try:
            film_type, film_preset_key = gen.resolve_film_type(
                film_type_preset=film_type_key, film_type_custom=film_type_custom
            )
            scene_count = (
                int(args.scene_count)
                if args.scene_count is not None
                else gen.DEFAULT_SCENE_COUNT
            )
            out_csv, story_title, _warnings = gen.generate_csv(
                topic=topic,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                stream=(not args.no_stream),
                film_type=film_type,
                film_preset_key=film_preset_key,
                scene_count=scene_count,
                background=background_text,
                out_path=out_path,
            )
            ws.cell(row=row_idx, column=args.write_title_col).value = story_title
            processed += 1
            print(f"Saved CSV: {out_csv}")
            print(f"Story title: {story_title}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] Row {row_idx} failed: {e}", file=sys.stderr)

    xlsx_out = args.xlsx_out or args.xlsx
    if not args.no_backup and os.path.abspath(xlsx_out) == os.path.abspath(args.xlsx):
        base, ext = os.path.splitext(args.xlsx)
        backup_path = f"{base}.bak_{ts}{ext}"
        try:
            import shutil

            shutil.copy2(args.xlsx, backup_path)
            print(f"\nBackup XLSX: {backup_path}")
        except Exception as e:
            print(f"\n[WARN] Failed to create backup XLSX: {e}", file=sys.stderr)

    wb.save(xlsx_out)
    print(f"\nSaved XLSX: {xlsx_out}")
    print(f"Done. ok={processed} failed={failed} out_dir={args.out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
