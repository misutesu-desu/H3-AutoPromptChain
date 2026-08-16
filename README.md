# H3 Auto Prompt Chain

A ComfyUI output node for unattended MiniMax H3 long-video generation from a list of prompts. It continues directly from saved audiovisual latents, supports per-shot generation budgets, can resume interrupted chains, and performs one final stitched decode.

## Requirements

- A current ComfyUI installation with MiniMax H3 support
- [Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)
- A compatible `MODEL`, `CLIP`, video `VAE`, audio `VAE`, and `SAMPLER`

No additional Python packages are required.

## Installation

Clone the repository into `ComfyUI/custom_nodes`, then restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/misutesu-desu/H3-AutoPromptChain.git
```

The node appears as **H3 Auto Prompt Chain v2** under **MiniMax H3/automation**.

## Shot prompts

Put one prompt on each line, or separate multiline prompts with a line containing `---`.

```text
[BALANCED] SHOT 01: A woman walks toward the apartment window.
---
[FAST] SHOT 02: She looks down at the street.
---
[MOTION][dur=10][tail=7] SHOT 03: She turns and runs toward the table.
```

Leading tags can override the defaults for each shot:

- `[FAST]`: 4 steps, 5 context frames
- `[BALANCED]` or `[NORMAL]`: 6 steps, 22 context frames
- `[QUALITY]` or `[HQ]`: 8 steps, 22 context frames
- `[MOTION]` or `[ACTION]`: at least 6 steps, 22 context frames
- `[dur=8]`: duration from 5 to 15 seconds
- `[steps=6]`: 1 to 40 sampling steps
- `[ctx=22]`: 5, 22, or 39 context frames
- `[tail=7]`: 0 to 68 landing-tail frames

For a fresh run, leave `start_clip` at `1`. If clip N fails after earlier latents were saved, set `start_clip` to N and keep the same prompt list, seed, and latent prefix.

## Outputs

- `video_path`: final stitched video path
- `run_info`: timing and completion summary
- `profile_path`: JSON profile with per-shot conditioning and sampling times

Generated latents, videos, and profiles stay inside ComfyUI's configured output directory.

## Notes


- Up to 80 shots are accepted in one unattended run.
- The final clip is saved without trimming its ending.
