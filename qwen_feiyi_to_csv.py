import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


DEFAULT_SCENE_COUNT = 50
DEFAULT_API_KEY_FILE_CANDIDATES = ("dashscope_api_key.txt", "secrets.json")

FILM_TYPE_PRESETS: Dict[str, str] = {
    "documentary": "纪录片",
    "microfilm": "微电影",
    "commercial": "广告片",
    "animation": "动画片",
    "narrative": "叙事短片",
    "poetic": "诗意短片",
    "music": "音乐短片",
}

FILM_TYPE_GUIDES: Dict[str, Dict[str, str]] = {
    "documentary": {
        "story": "类型要求：纪录片宣传短片。语气克制、信息密度高，尽量基于可核验的史实/机构信息，不夸张、不虚构人物经历。",
        "scene": "镜头风格：写实纪实；可包含：环境空镜、手部特写、工序细节、传承人采访式构图、档案/老照片质感、现场声氛围。",
    },
    "microfilm": {
        "story": "类型要求：微电影。以一个小人物/小事件切入，三幕式清晰（引子-转折-收束），情绪推进明确，结尾落回非遗价值与传承。",
        "scene": "镜头风格：电影化叙事；强调：人物动机、情绪层次、关键动作与道具、光影与景别变化。保持画面真实可拍。",
    },
    "commercial": {
        "story": "类型要求：广告片。节奏紧凑、卖点明确（文化价值、工艺独特性、当代意义、参与方式），更强调“记忆点/口号式”段落。",
        "scene": "镜头风格：高对比、高节奏；可包含：快速蒙太奇、文字信息叠加感（用画面描述）、强记忆点画面、收束到行动号召。",
    },
    "animation": {
        "story": "类型要求：动画片。允许更具想象力的隐喻与转场，但内容仍需围绕真实非遗信息与工艺逻辑，不要科幻设定喧宾夺主。",
        "scene": "镜头风格：动画化（2D/3D/水墨/剪纸等可选）；强调：形变转场、象征意象、材质笔触、色彩方案与节奏。",
    },
    "narrative": {
        "story": "类型要求：叙事短片。以“人物+选择”为主线，冲突/悬念在中段出现，结尾给出解决与传承的意义回响。",
        "scene": "镜头风格：叙事优先；强调：时间推进、因果关系、关键节点镜头（建立-冲突-高潮-余韵）。",
    },
    "poetic": {
        "story": "类型要求：诗意短片。语言更凝练、有意象与节奏感；允许留白与象征，但必须让观众理解非遗是什么、为什么重要。",
        "scene": "镜头风格：诗性表达；强调：意象并置、慢节奏、自然元素呼应、质感与光影、低饱和或统一色调、含蓄转场。",
    },
    "music": {
        "story": "类型要求：音乐短片（MV式结构）。以音乐节拍推动叙事与情绪，段落对应：引子-主歌-副歌-桥段-尾声（可简化）。旁白更短、更有节奏，强调意象与氛围，但仍需清楚点出非遗的名称、核心工艺与传承意义。",
        "scene": "镜头风格：MV化剪辑；强调：节拍点切换、动作对拍（敲打/穿针/落刀等）、重复意象、快慢镜头交替、镜头运动（推拉摇移/手持）与光影律动。画面描述避免出现具体歌曲名称与歌词引用。",
    },
}


STORY_TEMPLATE = (
    "你是一名专业的非遗策划与编导。\n"
    "请先搜索整理该非遗主题的官方资料（如：保护单位、级别、分布地区、代表性传承人、核心工艺/流程、历史沿革、现状与保护举措），并用简洁准确的方式介绍该非遗项目。\n"
    "{film_guide}\n"
    "然后撰写一个关于该主题的三分钟{film_type}脚本：聚焦历史文化与当代价值，结构清晰，有明确标题。\n"
    "约束：不编造不可证实的具体人名/机构/数据；不出现对话台词（如需表达观点用旁白）。"
)


def _build_scenes_template(
    *, scene_count: int, film_type: str, scene_guide: Optional[str]
) -> str:
    extra = f"补充风格要求：{scene_guide}\n" if scene_guide else ""
    return (
        f"作为一名专业短片导演，请使用以下格式创作{scene_count}个剧本场景：\n"
        f"短片类型：{film_type}\n"
        f"{extra}"
        "场景：（场景编号）\n"
        "旁白：{故事中的确切文本}\n"
        f"图像提示：{{以{film_type}的风格描述场景的详细文本到图像提示}}\n"
        f"视频提示:{{以{film_type}的风格描述场景的详细图像生成视频的提示}}\n"
    )


@dataclass(frozen=True)
class Scene:
    scene_number: int
    image_prompt: str
    video_prompt: str
    voiceover: Optional[str] = None


def _build_client(api_key: Optional[str], base_url: str) -> OpenAI:
    resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY") or _load_api_key_default()
    if not resolved_key:
        raise SystemExit(
            "Missing API key. Set env var DASHSCOPE_API_KEY, pass --api-key, or create dashscope_api_key.txt."
        )
    return OpenAI(api_key=resolved_key, base_url=base_url)


def _load_api_key_default() -> Optional[str]:
    for name in DEFAULT_API_KEY_FILE_CANDIDATES:
        path = os.path.join(os.path.dirname(__file__), name)
        key = _load_api_key_from_file(path)
        if key:
            return key
    return None


def _load_api_key_from_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        if path.lower().endswith(".json"):
            import json

            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                key = obj.get("DASHSCOPE_API_KEY") or obj.get("api_key") or obj.get("apiKey")
                return str(key).strip() if key else None
            return None

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text or None
    except Exception:
        return None

def _chat_text(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, str]],
    stream: bool,
) -> str:
    if stream:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        chunks: List[str] = []
        for chunk in completion:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                chunks.append(content)
                print(content, end="", flush=True)
        print("")
        return "".join(chunks).strip()

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()


def generate_story(
    client: OpenAI,
    *,
    model: str,
    topic: str,
    film_type: str,
    film_guide: Optional[str],
    background: Optional[str],
    stream: bool,
) -> str:
    background = (background or "").strip()
    background_block = (
        f"\n\n已提供背景资料（请优先使用；若不足再补充搜索，且不要编造超出资料范围的具体细节）：\n{background}"
        if background
        else ""
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                f"{STORY_TEMPLATE.format(film_type=film_type, film_guide=(film_guide or ''))}\n\n"
                f"主题：{topic}{background_block}\n\n输出要求：只输出故事正文（含标题）。"
            ),
        },
    ]
    return _chat_text(client, model=model, messages=messages, stream=stream)


def extract_story_title(story: str) -> str:
    for raw_line in (story or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*#{1,6}\s*", "", line).strip()
        line = re.sub(r"^\s*(标题|题目|片名)\s*[:：]\s*", "", line).strip()
        m = re.match(r"^《(.+?)》\s*$", line)
        if m:
            line = m.group(1).strip()
        line = line.strip("“”\"' ")
        return line
    return ""


def resolve_film_type(
    *, film_type_preset: Optional[str], film_type_custom: Optional[str]
) -> Tuple[str, Optional[str]]:
    custom = str(film_type_custom or "").strip()
    if custom:
        return custom, None
    if film_type_preset:
        return FILM_TYPE_PRESETS[film_type_preset], film_type_preset
    return FILM_TYPE_PRESETS["documentary"], "documentary"


def generate_csv(
    *,
    topic: str,
    model: str,
    base_url: str,
    api_key: Optional[str],
    stream: bool,
    film_type: str,
    film_preset_key: Optional[str],
    scene_count: int,
    background: Optional[str],
    out_path: Optional[str],
) -> Tuple[str, str, str, List[str]]:
    film_guide = None
    scene_guide = None
    if film_preset_key and film_preset_key in FILM_TYPE_GUIDES:
        film_guide = FILM_TYPE_GUIDES[film_preset_key].get("story")
        scene_guide = FILM_TYPE_GUIDES[film_preset_key].get("scene")

    client = _build_client(api_key, base_url)

    story = generate_story(
        client,
        model=model,
        topic=topic,
        film_type=film_type,
        film_guide=film_guide,
        background=background,
        stream=stream,
    )
    if not story:
        raise RuntimeError("Failed to generate story.")

    payload = generate_scenes_json(
        client,
        model=model,
        story=story,
        scene_count=scene_count,
        film_type=film_type,
        scene_guide=scene_guide,
        stream=stream,
    )
    scenes = scenes_from_payload(payload, expected_count=scene_count)
    warnings = validate_voiceovers_are_substrings(story, scenes)

    resolved_out_path = out_path or _default_out_path(topic)
    write_csv(resolved_out_path, scenes)

    title = extract_story_title(story) or topic
    return resolved_out_path, title, story, warnings


def generate_scenes_json(
    client: OpenAI,
    *,
    model: str,
    story: str,
    scene_count: int,
    film_type: str,
    scene_guide: Optional[str],
    stream: bool,
) -> Dict[str, Any]:
    system = "You are a helpful assistant."
    scenes_template = _build_scenes_template(
        scene_count=scene_count, film_type=film_type, scene_guide=scene_guide
    )
    user = (
        f"{scenes_template}\n"
        "请严格输出 JSON（不要 Markdown、不要代码块），结构如下：\n"
        '{\n  "scenes": [\n'
        '    {"scene": 1, "voiceover": "...", "image_prompt": "...", "video_prompt": "..."},\n'
        "    ...\n"
        "  ]\n}\n\n"
        "硬性要求：\n"
        f"1) 必须恰好 {scene_count} 个场景，scene 从 1 到 {scene_count}。\n"
        "2) voiceover 必须是“故事”中的原文片段（逐字一致），每个场景一个片段。\n"
        f"3) image_prompt / video_prompt 必须是{film_type}风格，细节充分（地点、人物、光线、镜头语言等）。\n"
        "4) 不要新增任何对话。\n\n"
        f"故事如下：\n{story}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw = _chat_text(client, model=model, messages=messages, stream=stream)
    return _parse_json_lenient(raw)


def _parse_json_lenient(text: str) -> Dict[str, Any]:
    try:
        import json

        return json.loads(text)
    except Exception:
        pass

    # Fallback: try to extract the largest {...} block.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("Model did not return JSON.")
    import json

    return json.loads(m.group(0))


def scenes_from_payload(payload: Dict[str, Any], *, expected_count: int) -> List[Scene]:
    if not isinstance(payload, dict) or "scenes" not in payload:
        raise ValueError("Invalid JSON: missing 'scenes'.")
    scenes = payload["scenes"]
    if not isinstance(scenes, list):
        raise ValueError("Invalid JSON: 'scenes' must be a list.")
    if len(scenes) != expected_count:
        raise ValueError(f"Expected {expected_count} scenes, got {len(scenes)}.")

    parsed: List[Scene] = []
    for item in scenes:
        if not isinstance(item, dict):
            raise ValueError("Invalid scene item (not an object).")
        scene_number = int(item.get("scene"))
        image_prompt = str(item.get("image_prompt") or "").strip()
        video_prompt = str(item.get("video_prompt") or "").strip()
        voiceover = (item.get("voiceover") or None)
        if voiceover is not None:
            voiceover = str(voiceover).strip()
        if not (1 <= scene_number <= expected_count):
            raise ValueError(f"Scene number out of range: {scene_number}.")
        if not image_prompt or not video_prompt:
            raise ValueError(f"Scene {scene_number} missing prompts.")
        parsed.append(
            Scene(
                scene_number=scene_number,
                image_prompt=image_prompt,
                video_prompt=video_prompt,
                voiceover=voiceover,
            )
        )

    parsed.sort(key=lambda s: s.scene_number)
    expected = list(range(1, expected_count + 1))
    got = [s.scene_number for s in parsed]
    if got != expected:
        raise ValueError(
            f"Scene numbering must be 1..{expected_count} in order. Got: {got}"
        )
    return parsed


def validate_voiceovers_are_substrings(story: str, scenes: List[Scene]) -> List[str]:
    warnings: List[str] = []
    for s in scenes:
        if not s.voiceover:
            warnings.append(f"Scene {s.scene_number}: voiceover missing.")
            continue
        if s.voiceover not in story:
            warnings.append(
                f"Scene {s.scene_number}: voiceover is not an exact substring of story."
            )
    return warnings


def write_csv(path: str, scenes: List[Scene]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["场景编号","对白", "图像提示", "视频提示"])
        for s in scenes:
            w.writerow([s.scene_number, s.voiceover, s.image_prompt, s.video_prompt])


def _default_out_path(topic: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", topic).strip("_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", f"qwen_scenes_{safe}_{ts}.csv")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a heritage story + scenes via Qwen, then export CSV."
    )
    p.add_argument("--topic", help="用户输入主题（中文/英文均可）")
    p.add_argument(
        "--film-type",
        choices=sorted(FILM_TYPE_PRESETS.keys()),
        default=None,
        help="短片类型（可选；不填则运行时选择）",
    )
    p.add_argument(
        "--film-type-custom",
        default=None,
        help="自定义短片类型（覆盖 --film-type），例如：'国风动画短片'、'公益广告片'",
    )
    p.add_argument(
        "--scene-count",
        type=int,
        default=DEFAULT_SCENE_COUNT,
        help=f"场景数量（默认 {DEFAULT_SCENE_COUNT}）",
    )
    p.add_argument(
        "--background",
        default=None,
        help="背景资料（可选；会优先用于生成故事脚本）",
    )
    p.add_argument(
        "--background-file",
        default=None,
        help="背景资料文件路径（UTF-8；覆盖 --background）",
    )
    p.add_argument("--model", default="qwen3-max", help="Qwen model name")
    p.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="DashScope OpenAI-compatible base URL",
    )
    p.add_argument("--api-key", help="API Key（可选，默认读 DASHSCOPE_API_KEY）")
    p.add_argument(
        "--api-key-file",
        default=None,
        help="API Key 文件路径（可选；读取纯文本 key 或 secrets.json；覆盖环境变量）",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    p.add_argument(
        "--out",
        help="输出 CSV 路径（默认 outputs/<topic>_<timestamp>.csv）",
    )
    args = p.parse_args(argv)

    topic = args.topic
    if not topic:
        topic = input("请输入主题：").strip()
    if not topic:
        print("Empty topic.", file=sys.stderr)
        return 2

    film_type: str
    film_preset_key: Optional[str] = None
    custom_type = str(args.film_type_custom or "").strip()
    if custom_type:
        film_type, film_preset_key = resolve_film_type(
            film_type_preset=None, film_type_custom=custom_type
        )
    elif args.film_type:
        film_type, film_preset_key = resolve_film_type(
            film_type_preset=args.film_type, film_type_custom=None
        )
    else:
        if not sys.stdin.isatty():
            film_type, film_preset_key = resolve_film_type(
                film_type_preset="documentary", film_type_custom=None
            )
        else:
            print("\n请选择短片类型（回车默认：纪录片）：")
            options = list(FILM_TYPE_PRESETS.items())
            for i, (k, label) in enumerate(options, start=1):
                print(f"{i}) {label}  (--film-type {k})")
            raw = input("输入序号：").strip()
            if raw:
                try:
                    idx = int(raw)
                    chosen_key, chosen_label = options[idx - 1]
                    film_type, film_preset_key = resolve_film_type(
                        film_type_preset=chosen_key, film_type_custom=None
                    )
                except Exception:
                    film_type, film_preset_key = resolve_film_type(
                        film_type_preset="documentary", film_type_custom=None
                    )
            else:
                film_type, film_preset_key = resolve_film_type(
                    film_type_preset="documentary", film_type_custom=None
                )

    scene_count = int(args.scene_count)
    if scene_count <= 0:
        print("--scene-count must be > 0", file=sys.stderr)
        return 2

    background: Optional[str] = None
    if args.background_file:
        with open(args.background_file, "r", encoding="utf-8") as f:
            background = f.read()
    elif args.background:
        background = str(args.background)

    api_key = args.api_key
    if args.api_key_file:
        api_key = _load_api_key_from_file(args.api_key_file) or api_key
    stream = not args.no_stream

    try:
        print("\n=== 生成中 ===")
        out_path, story_title, _story, warnings = generate_csv(
            topic=topic,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            stream=stream,
            film_type=film_type,
            film_preset_key=film_preset_key,
            scene_count=scene_count,
            background=background,
            out_path=args.out,
        )
    except Exception as e:
        print(f"Generation failed: {e}", file=sys.stderr)
        return 1

    if warnings:
        print("\n[WARN] Voiceover validation:")
        for w in warnings:
            print(f"- {w}")

    print(f"\nStory title: {story_title}")
    print(f"Saved CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
