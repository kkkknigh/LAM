# Text Edit MVP

This folder adds a minimal **text -> FLAME parameter edit** prototype for LAM.

## What it does

In `app_lam.py`, the Gradio UI now has:

- `Text Edit Prompt / 文本编辑指令`
- `Edit Strength / 编辑强度`

When you click **Generate**, the prompt is parsed into simple edit intents and applied to the driving FLAME sequence before `lam.infer_single_view(...)`.

Supported first-pass commands include:

- `笑一点`, `微笑`, `smile`
- `张嘴`, `嘴巴张开`, `mouth open`
- `闭嘴`, `mouth close`
- `眨眼`, `blink`
- `抬眉`, `挑眉`, `brow up`
- `皱眉`, `生气`, `frown`
- `看左`, `看右`, `看上`, `看下`
- `头左`, `头右`
- `脸瘦一点`, `下巴尖一点`, `鼻梁高一点`

## Important note

The FLAME expression basis is not perfectly semantic. The edit directions in
`flame_text_editor.py` are conservative default guesses. After you can run the app,
calibrate the indices and coefficients in `DIRECTION_TABLE` by visual testing.

The goal of this MVP is to prove the full path:

```text
text prompt -> parsed edit intent -> FLAME delta -> LAM rendering change
```
