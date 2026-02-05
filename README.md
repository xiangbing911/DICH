# Digital Intangible Cultural Heritage - Qwen 生成剧本场景并导出 CSV

## 功能
- 用户输入“主题”
- 调用千问（DashScope OpenAI 兼容接口）生成三分钟非遗宣传短片故事（聚焦人与自然、含标题、无对话）
- 基于故事生成场景（默认 50 个；风格随短片类型变化，含图像/视频提示）
- 最终导出 CSV：`场景编号, 图像提示, 视频提示`

## 环境准备
1) 安装依赖：
```bash
pip install openai openpyxl
```

2) 设置 API Key（推荐环境变量）：
```powershell
$env:DASHSCOPE_API_KEY="sk-xxx"
```

也可以在项目目录创建 `dashscope_api_key.txt`（只放一行 key，已加入 `.gitignore`），代码会自动读取。

## 运行
```powershell
python .\qwen_feiyi_to_csv.py --topic "你的主题"
```

默认会将 CSV 保存到 `outputs/` 目录下。

常用参数：
- `--no-stream`：关闭流式输出（更利于日志/调试）
- `--out path\to\file.csv`：指定输出路径
- `--film-type documentary|microfilm|commercial|animation|narrative|poetic|music`：选择短片类型
- `--film-type-custom "自定义类型"`：自定义短片类型（覆盖 `--film-type`）
- `--scene-count 50`：场景数量（默认 50）
- `--background "..."` / `--background-file path.txt`：提供背景资料（优先用于生成故事）
- `--model qwen3-max`：指定模型（默认 `qwen3-max`）
- `--base-url https://dashscope.aliyuncs.com/compatible-mode/v1`：指定接口地址
- `--api-key sk-xxx`：不使用环境变量时直接传入 key

## 批量处理 XLSX
`非遗项目汇总.xlsx` 三列：`项目名称 / 短片类型 / 背景资料`，逐行调用生成 CSV：
```powershell
python .\feiyi_xlsx_batch.py --xlsx ".\非遗项目汇总.xlsx"
```
