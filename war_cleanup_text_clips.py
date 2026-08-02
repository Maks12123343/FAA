"""
Remove text-heavy war clips from the active library index using Qwen2.5-VL.

This script is meant to run after video production finishes. It reads the active
war index, checks three frames per clip, moves bad clips to quarantine, writes a
backup index, updates index.json without the bad clips, and keeps resume state.

Default delete rule:
  should_delete == true
  confidence >= 0.85
  severity in {"medium", "heavy"}

Recommended server run:
  cd /workspace/FAA
  nohup /venv/main/bin/python war_cleanup_text_clips.py > text_cleanup.log 2>&1 &

Dry runs write separate dry-run state/report files so they do not affect the
real cleanup resume state.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import config


NICHE_NAME = "russia_ukraine_war"
DEFAULT_MODEL_PATH = os.environ.get("FAA_WAR_QWEN_MODEL", "/workspace/models/qwen2.5-vl-3b")
DEFAULT_INDEX = os.path.join(config.get_movies_dir(), NICHE_NAME, "index.json")
DEFAULT_FRAME_RATIOS = (0.10, 0.50, 0.90)
DEFAULT_MIN_CONFIDENCE = 0.85
DELETE_SEVERITIES = {"medium", "heavy"}
SAVE_EVERY = 20


TEXT_CLEANUP_PROMPT = """
You are cleaning a reusable war-footage clip library for YouTube montage.
You see 3 frames from ONE short clip: 10%, 50%, and 90% of the clip.

Your task is to decide whether this clip should be REMOVED from the reusable
library because visible text makes it too specific, ugly, or unsuitable for
generic montage.

Delete only obvious bad media:
- large subtitles or captions
- title cards or text-only screens
- news lower-third banners
- large centered text
- mirrored/reversed text
- phone/computer UI, social-media UI, maps with labels, HUD/UI overlays
- big watermarks/logos
- text dominating the frame or visible across multiple frames

Do NOT delete for harmless small text:
- tiny unreadable background text
- small vehicle numbers or markings
- small road signs
- tiny corner watermark that does not distract
- one small incidental label that does not dominate the clip

Severity definitions:
- none: no meaningful text
- small: tiny/incidental text, keep
- medium: noticeable text that hurts reuse, delete
- heavy: subtitles/title cards/UI/large text, delete

Return JSON only:
{
  "has_text": true,
  "severity": "none|small|medium|heavy",
  "text_type": "none|subtitles|title_card|watermark|ui_overlay|news_lower_third|mirrored_text|map_labels|other",
  "should_delete": true,
  "confidence": 0.0,
  "reason": "short factual reason based only on visible frames"
}

Be conservative about deleting normal footage, but aggressive about removing
subtitles, title cards, lower thirds, mirrored text, UI, and large overlay text.
""".strip()


def _utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _atomic_write_json(path: str, data, indent: int | None = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: str, item: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_frame_ratios(value: str) -> tuple[float, ...]:
    ratios = []
    for part in (value or "").split(","):
        try:
            ratio = float(part.strip())
        except ValueError:
            continue
        if 0.0 <= ratio <= 1.0:
            ratios.append(ratio)
    return tuple(ratios) or DEFAULT_FRAME_RATIOS


def _safe_relpath(path: str, root: str) -> str:
    try:
        rel = os.path.relpath(path, root)
        if rel.startswith(".."):
            raise ValueError
        return rel
    except Exception:
        return os.path.basename(path)


def _unique_dest(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while True:
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _backup_index(index_path: str, backup_dir: str) -> str:
    os.makedirs(backup_dir, exist_ok=True)
    dst = os.path.join(backup_dir, f"index.backup.text_cleanup.{_utc_stamp()}.json")
    shutil.copy2(index_path, dst)
    return dst


def _load_model(model_path: str):
    print(f"[cleanup] Loading Qwen2.5-VL model: {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[cleanup] Model ready. VRAM allocated={used:.1f} GB reserved={reserved:.1f} GB", flush=True)
    return model, processor


def _duration(path: str) -> float:
    try:
        proc = subprocess.run(
            [config.FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True,
            text=True,
            timeout=12,
        )
        return max(0.5, float(json.loads(proc.stdout)["format"]["duration"]))
    except Exception:
        return 3.0


def _extract_frame(path: str, timestamp: float, scale_width: int) -> str | None:
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        subprocess.run(
            [
                config.FFMPEG,
                "-y",
                "-ss",
                f"{timestamp:.2f}",
                "-i",
                path,
                "-frames:v",
                "1",
                "-vf",
                f"scale={scale_width}:-2",
                "-q:v",
                "3",
                tmp,
            ],
            capture_output=True,
            timeout=45,
        )
        if os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
            return tmp
    except Exception:
        pass
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return None


def _extract_frames(path: str, ratios: tuple[float, ...], scale_width: int) -> list[str]:
    dur = _duration(path)
    frames = []
    for ratio in ratios:
        ts = min(max(dur * ratio, 0.05), max(dur - 0.05, 0.05))
        frame = _extract_frame(path, ts, scale_width)
        if frame:
            frames.append(frame)
    return frames


def _parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        data = json.loads(match.group() if match else cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_decision(raw: dict | None, generated_text: str) -> dict:
    raw = raw or {}
    severity = str(raw.get("severity") or "none").strip().lower()
    if severity not in {"none", "small", "medium", "heavy"}:
        severity = "none"
    text_type = str(raw.get("text_type") or "none").strip().lower()
    text_type = re.sub(r"[^a-z0-9_]+", "_", text_type).strip("_") or "other"
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    has_text = bool(raw.get("has_text", severity in {"small", "medium", "heavy"}))
    should_delete = bool(raw.get("should_delete", False))
    reason = str(raw.get("reason") or generated_text[:300] or "").strip()
    return {
        "has_text": has_text,
        "severity": severity,
        "text_type": text_type,
        "should_delete": should_delete,
        "confidence": confidence,
        "reason": reason[:500],
    }


def _analyze_clip(path: str, model, processor, ratios: tuple[float, ...], max_new_tokens: int, scale_width: int) -> dict:
    frames = _extract_frames(path, ratios, scale_width)
    if not frames:
        return {
            "has_text": False,
            "severity": "none",
            "text_type": "none",
            "should_delete": False,
            "confidence": 0.0,
            "reason": "Could not extract frames; kept by default.",
            "error": "no_frames",
        }

    try:
        images = []
        for frame in frames:
            with Image.open(frame) as img:
                images.append(img.convert("RGB"))

        labels = ("FRAME 10 PERCENT", "FRAME 50 PERCENT", "FRAME 90 PERCENT")
        content = []
        for i, img in enumerate(images):
            content.append({"type": "text", "text": labels[i] if i < len(labels) else f"FRAME {i + 1}"})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": TEXT_CLEANUP_PROMPT})

        messages = [{"role": "user", "content": content}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        generated_text = processor.decode(generated, skip_special_tokens=True).strip()
        return _normalize_decision(_parse_json_object(generated_text), generated_text)
    except Exception as exc:
        print(f"[cleanup] Analyze error for {os.path.basename(path)}: {exc}", flush=True)
        return {
            "has_text": False,
            "severity": "none",
            "text_type": "none",
            "should_delete": False,
            "confidence": 0.0,
            "reason": f"Analyze error; kept by default: {exc}",
            "error": str(exc),
        }
    finally:
        for frame in frames:
            try:
                os.unlink(frame)
            except OSError:
                pass


def _should_quarantine(decision: dict, min_confidence: float, severities: set[str]) -> bool:
    return (
        bool(decision.get("should_delete"))
        and float(decision.get("confidence") or 0.0) >= min_confidence
        and str(decision.get("severity") or "").lower() in severities
    )


def _move_to_quarantine(path: str, library_root: str, quarantine_dir: str) -> str:
    rel = _safe_relpath(path, library_root)
    dst = _unique_dest(os.path.join(quarantine_dir, rel))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(path, dst)
    return dst


def _write_filtered_index(index_path: str, original_index: dict, active_clips: list[dict], cleanup_meta: dict) -> None:
    updated = dict(original_index)
    updated["clips"] = active_clips
    updated["total_dur"] = round(sum(float(c.get("duration", 3.0)) for c in active_clips), 2)
    updated["text_cleanup"] = cleanup_meta
    _atomic_write_json(index_path, updated, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Quarantine text-heavy war clips and update index.json.")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--niche", default=NICHE_NAME)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--frames", default=",".join(str(x) for x in DEFAULT_FRAME_RATIOS))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--delete-severities", default="medium,heavy")
    parser.add_argument("--max-clips", type=int, default=0, help="Debug limit. 0 means all active clips.")
    parser.add_argument("--force", action="store_true", help="Ignore previous cleanup state.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report, but do not move files or rewrite index.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--scale-width", type=int, default=896)
    parser.add_argument("--library-root", default=os.path.join(config.LIBRARY_DIR, NICHE_NAME))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = os.path.abspath(args.index)
    index_dir = os.path.dirname(index_path)
    state_name = "text_cleanup_dry_run_state.json" if args.dry_run else "text_cleanup_state.json"
    report_name = "text_cleanup_dry_run_report.jsonl" if args.dry_run else "text_cleanup_report.jsonl"
    state_path = os.path.join(index_dir, state_name)
    report_path = os.path.join(index_dir, report_name)
    quarantine_dir = os.path.join(index_dir, "quarantine_text_bad")
    backup_dir = os.path.join(index_dir, "backups")
    ratios = _parse_frame_ratios(args.frames)
    severities = {s.strip().lower() for s in args.delete_severities.split(",") if s.strip()}

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    original_index = _load_json(index_path, {})
    clips = [c for c in original_index.get("clips", []) if isinstance(c, dict)]
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    state = {"done": {}, "quarantined": [], "started_at": _utc_stamp()} if args.force else _load_json(
        state_path,
        {"done": {}, "quarantined": [], "started_at": _utc_stamp()},
    )
    state.setdefault("done", {})
    state.setdefault("quarantined", [])

    if not args.dry_run and not state.get("backup_index"):
        state["backup_index"] = _backup_index(index_path, backup_dir)
        _atomic_write_json(state_path, state, indent=2)
        print(f"[cleanup] Backup index: {state['backup_index']}", flush=True)

    print(f"[cleanup] Index: {index_path}", flush=True)
    print(f"[cleanup] Clips in active index: {len(clips)}", flush=True)
    print(f"[cleanup] Frames: {ratios}", flush=True)
    print(f"[cleanup] Delete rule: should_delete=true, confidence>={args.min_confidence}, severity in {sorted(severities)}", flush=True)
    print(f"[cleanup] Dry run: {args.dry_run}", flush=True)

    remaining = [c for c in clips if c.get("file") not in state["done"]]
    print(f"[cleanup] Already checked: {len(state['done'])}, remaining: {len(remaining)}", flush=True)
    if not remaining:
        print("[cleanup] Nothing to do.", flush=True)
        return 0

    model, processor = _load_model(args.model_path)
    start = time.time()
    checked_now = 0
    quarantined_now = 0
    errors_now = 0

    for idx, clip in enumerate(remaining, start=1):
        path = clip.get("file") or ""
        if not path or not os.path.exists(path):
            decision = {
                "has_text": False,
                "severity": "none",
                "text_type": "none",
                "should_delete": False,
                "confidence": 0.0,
                "reason": "File missing before cleanup; kept out of quarantine.",
                "error": "missing_file",
            }
            errors_now += 1
        else:
            decision = _analyze_clip(
                path,
                model,
                processor,
                ratios=ratios,
                max_new_tokens=args.max_new_tokens,
                scale_width=args.scale_width,
            )
            if decision.get("error"):
                errors_now += 1

        action = "keep"
        quarantine_path = ""
        if path and os.path.exists(path) and _should_quarantine(decision, args.min_confidence, severities):
            action = "quarantine"
            if not args.dry_run:
                quarantine_path = _move_to_quarantine(path, os.path.abspath(args.library_root), quarantine_dir)
                quarantined_now += 1

        record = {
            "checked_at": _utc_stamp(),
            "clip_id": clip.get("id") or os.path.splitext(os.path.basename(path))[0],
            "file": path,
            "action": action,
            "quarantine_path": quarantine_path,
            "decision": decision,
        }
        state["done"][path] = record
        if quarantine_path:
            state["quarantined"].append({"file": path, "quarantine_path": quarantine_path})
        _append_jsonl(report_path, record)
        checked_now += 1

        if idx % max(1, args.save_every) == 0 or idx == len(remaining):
            elapsed = max(0.001, time.time() - start)
            rate = checked_now / elapsed
            eta = (len(remaining) - idx) / rate if rate > 0 else 0.0
            if not args.dry_run:
                quarantined_files = {
                    item["file"] for item in state.get("quarantined", []) if item.get("file")
                }
                active_clips = [c for c in original_index.get("clips", []) if c.get("file") not in quarantined_files]
                cleanup_meta = {
                    "version": "war-text-cleanup-qwen-v1",
                    "updated_at": _utc_stamp(),
                    "min_confidence": args.min_confidence,
                    "delete_severities": sorted(severities),
                    "frames": list(ratios),
                    "checked_total": len(state["done"]),
                    "quarantined_total": len(state.get("quarantined", [])),
                    "report_path": report_path,
                    "state_path": state_path,
                    "quarantine_dir": quarantine_dir,
                    "backup_index": state.get("backup_index", ""),
                }
                _write_filtered_index(index_path, original_index, active_clips, cleanup_meta)
            _atomic_write_json(state_path, state, indent=2)
            print(
                f"[cleanup] {idx}/{len(remaining)} checked | quarantined_now={quarantined_now} "
                f"| errors_now={errors_now} | {rate:.3f} clips/s | ETA {eta/3600:.1f}h",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not args.dry_run:
        quarantined_files = {item["file"] for item in state.get("quarantined", []) if item.get("file")}
        active_clips = [c for c in original_index.get("clips", []) if c.get("file") not in quarantined_files]
        cleanup_meta = {
            "version": "war-text-cleanup-qwen-v1",
            "updated_at": _utc_stamp(),
            "min_confidence": args.min_confidence,
            "delete_severities": sorted(severities),
            "frames": list(ratios),
            "checked_total": len(state["done"]),
            "quarantined_total": len(state.get("quarantined", [])),
            "report_path": report_path,
            "state_path": state_path,
            "quarantine_dir": quarantine_dir,
            "backup_index": state.get("backup_index", ""),
        }
        _write_filtered_index(index_path, original_index, active_clips, cleanup_meta)

    elapsed = max(0.001, time.time() - start)
    print("[cleanup] DONE", flush=True)
    print(f"[cleanup] Checked this run: {checked_now}", flush=True)
    print(f"[cleanup] Quarantined this run: {quarantined_now}", flush=True)
    print(f"[cleanup] Errors this run: {errors_now}", flush=True)
    print(f"[cleanup] Elapsed: {elapsed/3600:.2f}h", flush=True)
    print(f"[cleanup] Report: {report_path}", flush=True)
    print(f"[cleanup] State: {state_path}", flush=True)
    if not args.dry_run:
        print(f"[cleanup] Quarantine: {quarantine_dir}", flush=True)
        print(f"[cleanup] Index updated: {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
