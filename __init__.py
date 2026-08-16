import json
import logging
import os
import re
import time

import comfy.samplers
import folder_paths
import nodes

from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    BasicScheduler,
    RandomNoise,
    SamplerCustomAdvanced,
)

_LOG = logging.getLogger("h3_auto_prompt_chain")


def _v3_arg(node_output, index=0):
    if hasattr(node_output, "args"):
        return node_output.args[index]
    if isinstance(node_output, (tuple, list)):
        return node_output[index]
    raise TypeError(f"Unexpected ComfyUI node output type: {type(node_output)!r}")


def _require_node(name):
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"H3 Auto Prompt Chain requires '{name}'. "
            "Install/update Herrgotts-H3-Infinite-Continuation-Suite and restart ComfyUI."
        )
    return cls


def _split_shots(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("shot_prompts is empty.")

    if re.search(r"(?m)^\s*---+\s*$", text):
        chunks = re.split(r"(?m)^\s*---+\s*$", text)
        shots = [c.strip() for c in chunks if c.strip()]
    else:
        shots = [
            ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]

    if not shots:
        raise ValueError("No shot prompts found.")
    if len(shots) > 80:
        raise ValueError(f"Maximum 80 shots per unattended run; got {len(shots)}.")
    return shots


_TAG_RE = re.compile(r"^\s*\[\s*([A-Za-z_]+)(?:\s*=\s*([^\]]+))?\s*\]\s*", re.I)


def _parse_shot(block, default_duration, default_steps, default_context, default_tail):
    """Parse optional leading shot-budget tags without exposing them to Qwen.

    Supported examples:
      [FAST]                        -> 4 steps, ctx 5
      [BALANCED]                    -> 6 steps, ctx 22
      [QUALITY]                     -> 8 steps, ctx 22
      [dur=10.1][steps=4][ctx=5]
      [dur=8][steps=6][ctx=22][tail=7]

    Tags are deliberately simple so another LLM can emit them reliably.
    """
    duration = float(default_duration)
    steps = int(default_steps)
    context = int(default_context)
    tail = int(default_tail)
    text = block
    tags = []

    while True:
        m = _TAG_RE.match(text)
        if not m:
            break
        key = m.group(1).strip().lower()
        value = (m.group(2) or "").strip()
        tags.append(m.group(0).strip())
        text = text[m.end():]

        if key == "fast":
            steps, context = 4, 5
        elif key in ("balanced", "normal"):
            steps, context = 6, 22
        elif key in ("quality", "hq"):
            steps, context = 8, 22
        elif key in ("motion", "action"):
            steps, context = max(6, steps), 22
        elif key in ("dur", "duration", "sec", "seconds"):
            duration = float(value)
        elif key in ("steps", "step"):
            steps = int(value)
        elif key in ("ctx", "context"):
            context = int(value)
        elif key in ("tail", "landing_tail"):
            tail = int(value)
        else:
            raise ValueError(f"Unknown shot tag [{key}{'=' + value if value else ''}]")

    body = text.strip()
    if not body:
        raise ValueError(f"Shot contains tags but no prompt text: {' '.join(tags)}")

    if not (5.0 <= duration <= 15.0):
        raise ValueError(f"Shot duration must be 5.0-15.0 seconds; got {duration}.")
    if not (1 <= steps <= 40):
        raise ValueError(f"Shot steps must be 1-40; got {steps}.")
    if context not in (5, 22, 39):
        raise ValueError(f"Shot ctx must be 5, 22, or 39; got {context}.")
    if not (0 <= tail <= 68):
        raise ValueError(f"Shot tail must be 0-68; got {tail}.")

    return {
        "prompt": body,
        "duration": duration,
        "steps": steps,
        "context": context,
        "tail": tail,
        "tags": tags,
    }


def _pixel_frame_count(latent):
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("Expected H3 LATENT with 'samples'.")
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(f"Unexpected H3 latent container: {type(samples)!r}")
    if not parts:
        raise ValueError("H3 latent has no video stream.")
    video = parts[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    t = int(video.shape[2])
    frame_per_token = (1, 4, 4, 4, 4)
    return sum(frame_per_token[k % 5] for k in range(t))


def _manual_handover(latent, ignored_tail_frames, requested_tail_frames):
    frame_count = _pixel_frame_count(latent)
    tail = max(0, min(int(ignored_tail_frames), frame_count - 1))
    end_frame = frame_count - tail - 1
    return {
        "available": True,
        "version": 100,
        "release_version": "auto-prompt-chain-2.0",
        "frame_count": frame_count,
        "freeze_detected": False,
        "detector_mode": "manual_no_decode",
        "no_lock_fallback_applied": True,
        "no_lock_fallback_requested_excluded_frames": int(requested_tail_frames),
        "handover_end_frame": end_frame,
        "ideal_handover_end_frame": end_frame,
        "phase_aligned_target_end_frame": end_frame,
        "landing_tail_frames": tail,
        "legacy_landing_tail_frames": tail,
        "safety_mode": "fixed",
        "confidence": 0.0,
    }


def _profile_path(latent_prefix):
    output_dir = folder_paths.get_output_directory()
    prefix = (latent_prefix or "h3_autochain/clip").replace("\\", "/")
    folder, _, _, _, _ = folder_paths.get_save_image_path(prefix, output_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(folder, f"profile_{stamp}.json")


class H3AutoPromptChainRunnerOptimal:
    """One-click H3 long-video runner optimized for low-VRAM unattended chains.

    Key speed choices:
    - direct AV-latent continuation (no intermediate VAE decode/re-encode)
    - caller-supplied sampler (recommended: Larry H3 Turbo sampler)
    - per-shot 4/6/8-step budgets and 5/22/39-frame context tags
    - scheduler sigma cache per step count
    - no per-shot gc.collect()
    - one memory-bounded final saved-chain stitch/decode
    - JSON profiler so the real bottleneck is measurable on the user's machine
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "sampler": ("SAMPLER",),
                "global_prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "GLOBAL CONTINUITY:\n"
                        "Preserve established character identity, facial features, body proportions, "
                        "wardrobe, scene geometry, object state, lighting logic and cinematic style across shots.\n"
                        "Natural temporal motion and coherent native audio.\n"
                        "non_diegetic_music: N/A"
                    ),
                }),
                "shot_prompts": ("STRING", {
                    "multiline": True,
                    "default": (
                        "[BALANCED] SHOT 01: A woman walks slowly through a quiet apartment toward the window.\n"
                        "---\n"
                        "[FAST] SHOT 02: She reaches the window and looks down at the street.\n"
                        "---\n"
                        "[MOTION] SHOT 03: She suddenly turns and walks quickly toward the table."
                    ),
                    "tooltip": "Separate multiline shots with ---. Optional leading tags: [FAST], [BALANCED], [QUALITY], [dur=10.1], [steps=4], [ctx=5], [tail=7].",
                }),
                "width": ("INT", {"default": 608, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 352, "min": 32, "max": 4096, "step": 32}),
                "default_duration_seconds": ("FLOAT", {"default": 8.0, "min": 5.0, "max": 15.0, "step": 0.1}),
                "default_steps": ("INT", {"default": 6, "min": 1, "max": 40}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "default_context_frames": (["5", "22", "39"], {"default": "22"}),
                "manual_tail_frames": ("INT", {
                    "default": 7, "min": 0, "max": 68, "step": 1,
                    "tooltip": "Fast no-decode handover tail. 7 mirrors the suite's balanced no-lock fallback request before phase alignment."
                }),
                "base_seed": ("INT", {"default": 123456789, "min": 0, "max": 0xffffffffffffffff}),
                "start_clip": ("INT", {
                    "default": 1, "min": 1, "max": 80,
                    "tooltip": "1=fresh run. If clip N failed, keep saved latents and resume with N."
                }),
                "latent_prefix": ("STRING", {"default": "h3_autochain/clip"}),
                "output_prefix": ("STRING", {"default": "video/H3_AUTO_CHAIN_FINAL"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
            },
            "optional": {
                "first_frame": ("IMAGE", {
                    "tooltip": "Optional first-frame anchor for clip 1. Leave disconnected for pure T2V."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "run_info", "profile_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/automation"
    DESCRIPTION = (
        "Optimized one-click H3 prompt-list chain: dynamic per-shot step/context budgets, "
        "direct AV latent continuation, resumable latent saves, timing profiler and one final memory-bounded decode."
    )

    def run(
        self,
        model,
        clip,
        video_vae,
        audio_vae,
        sampler,
        global_prompt,
        shot_prompts,
        width=608,
        height=352,
        default_duration_seconds=8.0,
        default_steps=6,
        scheduler="simple",
        default_context_frames="22",
        manual_tail_frames=7,
        base_seed=123456789,
        start_clip=1,
        latent_prefix="h3_autochain/clip",
        output_prefix="video/H3_AUTO_CHAIN_FINAL",
        crf=18,
        first_frame=None,
    ):
        output_dir = folder_paths.get_output_directory()
        folder_paths.get_save_image_path(latent_prefix, output_dir)
        folder_paths.get_save_image_path(output_prefix, output_dir)
        raw_shots = _split_shots(shot_prompts)
        shots = [
            _parse_shot(
                s,
                default_duration_seconds,
                default_steps,
                int(default_context_frames),
                manual_tail_frames,
            )
            for s in raw_shots
        ]
        total = len(shots)
        start_clip = int(start_clip)
        if start_clip > total:
            raise ValueError(f"start_clip={start_clip}, but prompt list has only {total} shots.")

        ContinueCls = _require_node("H3ContinuousContinueV11")
        SaveCls = _require_node("H3ContinuousSaveLatent")
        LoadCls = _require_node("H3ContinuousLoadLatent")
        StitchCls = _require_node("H3ContinuousStitchSavedChainV11")

        sigmas_cache = {}
        global_prompt = (global_prompt or "").strip()
        continuation_rule = (
            "CONTINUE the incoming audiovisual latent naturally. "
            "Do not reset identity, pose history, camera direction, object state, ambience, or voice."
        )

        prev_latent = None
        prev_head_context = 0
        profile = {
            "version": "2.0",
            "width": int(width),
            "height": int(height),
            "scheduler": scheduler,
            "start_clip": start_clip,
            "total_shots": total,
            "shots": [],
        }
        run_started = time.perf_counter()

        if start_clip > 1:
            t0 = time.perf_counter()
            loaded = LoadCls().load(latent_prefix, start_clip - 1)
            prev_latent = loaded[0]
            load_s = time.perf_counter() - t0
            profile["resume_load_seconds"] = load_s
            _LOG.info("H3 Auto Prompt Chain: resumed from saved clip %d in %.2fs", start_clip - 1, load_s)

        for clip_index in range(start_clip, total + 1):
            spec = shots[clip_index - 1]
            shot = spec["prompt"]
            duration = float(spec["duration"])
            steps = int(spec["steps"])
            context = int(spec["context"])
            tail = int(spec["tail"])
            shot_t0 = time.perf_counter()

            # Conditioning / continuation construction. This is where the 32B
            # prompt encoder cost lands; timed separately from diffusion sampling.
            cond_t0 = time.perf_counter()
            if clip_index == 1:
                prompt = "\n\n".join(x for x in (global_prompt, shot) if x)
                requested_frames = max(5, int(round(duration * 24.0)))
                result = MiniMaxH3ImageToVideo.execute(
                    clip,
                    video_vae,
                    prompt,
                    int(width),
                    int(height),
                    requested_frames,
                    first_frame=first_frame,
                    last_frame=None,
                )
                conditioning, latent_in = result.args
                current_head_context = 0
                ignored_prev_tail = 0
            else:
                if prev_latent is None:
                    raise RuntimeError("Missing previous latent for continuation.")
                prompt = "\n\n".join(
                    x for x in (global_prompt, continuation_rule, shot) if x
                )
                cont = ContinueCls().build(
                    clip,
                    video_vae,
                    prev_latent,
                    prompt,
                    int(width),
                    int(height),
                    duration,
                    context_frames=str(context),
                    handover_mode="manual",
                    alignment_mode="phase_aligned_extended",
                    manual_landing_tail_frames=tail,
                    ref_image_size="match",
                    handover=None,
                    last_frame=None,
                    reference_image=None,
                )
                conditioning, latent_in, current_head_context, ignored_prev_tail, _handover_info = cont

                # Only save clip N-1 once the exact cutoff used by clip N is known.
                # On the first resumed clip, do not overwrite the already-saved source.
                if clip_index > start_clip or start_clip == 1:
                    previous_index = clip_index - 1
                    handover = _manual_handover(prev_latent, ignored_prev_tail, tail)
                    SaveCls().save(
                        prev_latent,
                        latent_prefix,
                        previous_index,
                        handover=handover,
                        head_context_frames=int(prev_head_context),
                    )

            conditioning_s = time.perf_counter() - cond_t0

            # Scheduler output depends only on model/scheduler/step count, so cache
            # it for the whole unattended run instead of rebuilding every shot.
            if steps not in sigmas_cache:
                sigmas_cache[steps] = _v3_arg(
                    BasicScheduler.execute(model, scheduler, steps, 1.0)
                )
            sigmas = sigmas_cache[steps]

            seed = (int(base_seed) + clip_index - 1) & 0xFFFFFFFFFFFFFFFF
            guider = _v3_arg(BasicGuider.execute(model, conditioning))
            noise = _v3_arg(RandomNoise.execute(seed))

            sample_t0 = time.perf_counter()
            sampled = _v3_arg(
                SamplerCustomAdvanced.execute(
                    noise, guider, sampler, sigmas, latent_in
                )
            )
            sampling_s = time.perf_counter() - sample_t0

            prev_latent = sampled
            prev_head_context = int(current_head_context)
            shot_total_s = time.perf_counter() - shot_t0

            row = {
                "clip": clip_index,
                "duration_seconds_requested": duration,
                "steps": steps,
                "context_frames_requested": context,
                "manual_tail_frames": tail,
                "actual_head_context_frames": int(current_head_context),
                "ignored_previous_tail_frames": int(ignored_prev_tail),
                "seed": seed,
                "conditioning_seconds": round(conditioning_s, 3),
                "sampling_seconds": round(sampling_s, 3),
                "total_seconds": round(shot_total_s, 3),
            }
            profile["shots"].append(row)
            _LOG.info(
                "H3 Auto Prompt Chain: clip %d/%d | %.1fs | %d steps | ctx %d->%d | encode/cond %.2fs | sample %.2fs | total %.2fs",
                clip_index, total, duration, steps, context, int(current_head_context),
                conditioning_s, sampling_s, shot_total_s,
            )

        # Preserve the full ending of the final clip.
        save_t0 = time.perf_counter()
        SaveCls().save(
            prev_latent,
            latent_prefix,
            total,
            handover=None,
            head_context_frames=int(prev_head_context),
        )
        final_save_s = time.perf_counter() - save_t0

        stitch_t0 = time.perf_counter()
        video_path, stitch_info = StitchCls().stitch(
            video_vae,
            audio_vae,
            latent_prefix=latent_prefix,
            first_clip=1,
            last_clip=total,
            filename_prefix=output_prefix,
            video_crossfade_frames=4,
            audio_crossfade_ms=15.0,
            luminance_match=False,
            luminance_fade_frames=16,
            max_luminance_correction_percent=10.0,
            crf=int(crf),
            max_safe_tail_bridge_frames=0,
        )
        stitch_s = time.perf_counter() - stitch_t0
        total_s = time.perf_counter() - run_started

        profile["final_latent_save_seconds"] = round(final_save_s, 3)
        profile["final_stitch_decode_seconds"] = round(stitch_s, 3)
        profile["run_total_seconds"] = round(total_s, 3)
        profile["video_path"] = str(video_path)

        pp = _profile_path(latent_prefix)
        with open(pp, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        avg_sample = 0.0
        if profile["shots"]:
            avg_sample = sum(x["sampling_seconds"] for x in profile["shots"]) / len(profile["shots"])
        run_info = (
            f"Completed {total} shots | {width}x{height} | dynamic 4/6/8-style budgets supported | "
            f"scheduler={scheduler} | final={video_path}\n"
            f"Total {total_s:.1f}s | final stitch/decode {stitch_s:.1f}s | avg sampling/shot {avg_sample:.1f}s\n"
            f"Profiler: {pp}\n{stitch_info}"
        )
        return (video_path, run_info, pp)


NODE_CLASS_MAPPINGS = {
    "H3AutoPromptChainRunnerOptimal": H3AutoPromptChainRunnerOptimal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AutoPromptChainRunnerOptimal": "H3 Auto Prompt Chain v2",
}
