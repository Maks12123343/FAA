"""Safely classify and quarantine text-heavy clips with Qwen2.5-VL.

The script is deliberately two-phase:
1. Analyze every clip and persist a resumable decision journal.
2. Only after analysis and safety guards pass, annotate the active index and
   move clearly unsuitable clips into a recoverable quarantine directory.

Small or incidental text is kept and marked ``no_mirror`` so montage
uniqualization never reverses it. Analysis errors are also kept with
``no_mirror``. No library files are moved while analysis is still running.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter

import config


NICHE_NAME = "russia_ukraine_war"
SCRIPT_VERSION = "war-text-cleanup-qwen-v2"
DEFAULT_MODEL_PATH = os.environ.get("FAA_WAR_QWEN_MODEL", "/workspace/models/qwen2.5-vl-3b")
DEFAULT_INDEX = os.path.join(config.get_movies_dir(), NICHE_NAME, "index.json")
DEFAULT_FRAME_RATIOS = (0.10, 0.50, 0.90)
DEFAULT_PILOT_CLIPS = 300
DEFAULT_MAX_DELETE_RATE = 0.30
DEFAULT_MAX_ERROR_RATE = 0.02
DEFAULT_MIN_REMOVE_CONFIDENCE = 0.80
SAVE_EVERY = 20

TEXT_SIZES = ("none", "tiny", "small", "medium", "large", "dominant")
TEXT_SIZE_RANK = {name: rank for rank, name in enumerate(TEXT_SIZES)}
TEXT_TYPES = {
    "none",
    "subtitles",
    "title_card",
    "text_only_screen",
    "news_lower_third",
    "large_overlay",
    "ui_screen",
    "map_labels",
    "mirrored_text",
    "watermark_logo",
    "physical_label",
    "vehicle_marking",
    "road_sign",
    "other",
}
NATURAL_TEXT_TYPES = {"physical_label", "vehicle_marking", "road_sign"}
PROMINENT_TEXT_TYPES = {
    "subtitles",
    "title_card",
    "text_only_screen",
    "news_lower_third",
    "large_overlay",
    "ui_screen",
    "map_labels",
    "mirrored_text",
    "watermark_logo",
}


TEXT_CLEANUP_PROMPT = """
You are inspecting reusable war-footage clips for a YouTube montage library.
The three images are from ONE short clip at 10%, 50%, and 90%.

Classify visible text precisely. Do not decide deletion yourself. The software
will apply a conservative deterministic policy from your measurements.

Distinguish harmful editorial text from harmless natural text:
- subtitles: readable captions/dialogue near the top or bottom
- title_card: a title or headline is the main visual
- text_only_screen: article, document, slide, or mostly text
- news_lower_third: news banner/name bar/ticker
- large_overlay: large editorial words laid over normal footage
- ui_screen: phone, computer, website, social-media, HUD, or app interface
- map_labels: a map whose labels are visually important
- mirrored_text: clearly reversed readable text
- watermark_logo: channel logo or watermark
- physical_label: writing on clothing, helmet, patch, product, building, etc.
- vehicle_marking: vehicle number, registration, or equipment marking
- road_sign: a real sign in the filmed scene
- other: visible text that fits none of the above

For EACH frame estimate the largest text block:
- none: no visible text
- tiny: under about 2% of the frame; corner mark or barely readable detail
- small: about 2-5%; incidental and not distracting
- medium: about 5-12%; clearly readable and noticeable
- large: about 12-30%; visually prominent
- dominant: over 30% or the frame is mainly text

Important examples:
- A small logo, patch, cap label, hoodie word, vehicle number, or road sign is
  small natural text. Do not inflate its size because it is readable.
- One or two lines of persistent readable subtitles are at least medium.
- A full article, title card, laptop/phone interface, or large banner is large
  or dominant.
- If uncertain, choose the smaller size and lower confidence.

Return JSON only, exactly this shape:
{
  "text_type": "none|subtitles|title_card|text_only_screen|news_lower_third|large_overlay|ui_screen|map_labels|mirrored_text|watermark_logo|physical_label|vehicle_marking|road_sign|other",
  "frame_text_sizes": ["none", "none", "none"],
  "readable_text": false,
  "mirrored_text": false,
  "confidence": 0.0,
  "reason": "short factual description of the visible text"
}
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


def _backup_index(index_path: str, backup_dir: str, label: str = "text_cleanup") -> str:
    os.makedirs(backup_dir, exist_ok=True)
    dst = os.path.join(backup_dir, f"index.backup.{label}.{_utc_stamp()}.json")
    shutil.copy2(index_path, dst)
    return dst


def _archive_file(path: str, label: str) -> str:
    if not os.path.exists(path):
        return ""
    base, ext = os.path.splitext(path)
    dst = f"{base}.{label}.{_utc_stamp()}{ext}"
    os.replace(path, dst)
    return dst


def _index_fingerprint(clips: list[dict]) -> str:
    digest = hashlib.sha256()
    for path in sorted(str(c.get("file") or "") for c in clips):
        digest.update(path.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_model(model_path: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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


def _normalize_text_type(value) -> str:
    text_type = re.sub(r"[^a-z0-9_]+", "_", str(value or "none").strip().lower()).strip("_")
    aliases = {
        "ui_overlay": "ui_screen",
        "social_media_ui": "ui_screen",
        "map": "map_labels",
        "watermark": "watermark_logo",
        "logo": "watermark_logo",
        "vehicle_numbers": "vehicle_marking",
        "headband": "physical_label",
    }
    text_type = aliases.get(text_type, text_type)
    return text_type if text_type in TEXT_TYPES else "other"


def _normalize_size(value) -> str:
    size = str(value or "none").strip().lower()
    aliases = {"no": "none", "minimal": "tiny", "moderate": "medium", "heavy": "dominant"}
    size = aliases.get(size, size)
    return size if size in TEXT_SIZE_RANK else "none"


def _normalize_decision(raw: dict | None, generated_text: str, expected_frames: int) -> dict:
    if not isinstance(raw, dict):
        return _error_decision("invalid_json", "Model did not return valid JSON; clip kept without mirroring.")

    text_type = _normalize_text_type(raw.get("text_type"))
    sizes_raw = raw.get("frame_text_sizes")
    if not isinstance(sizes_raw, list):
        sizes_raw = []
    sizes = [_normalize_size(v) for v in sizes_raw[:expected_frames]]
    sizes.extend(["none"] * max(0, expected_frames - len(sizes)))

    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    readable = bool(raw.get("readable_text", False))
    mirrored = bool(raw.get("mirrored_text", False)) or text_type == "mirrored_text"
    frames_with_text = sum(1 for size in sizes if size != "none")
    max_size = max(sizes, key=lambda size: TEXT_SIZE_RANK[size], default="none")
    has_text = frames_with_text > 0 or readable or text_type != "none"
    if not has_text:
        text_type = "none"
        max_size = "none"

    return {
        "has_text": has_text,
        "text_type": text_type,
        "frame_text_sizes": sizes,
        "frames_with_text": frames_with_text,
        "max_text_size": max_size,
        "readable_text": readable,
        "mirrored_text": mirrored,
        "confidence": confidence,
        "reason": str(raw.get("reason") or generated_text[:300] or "").strip()[:500],
    }


def _error_decision(code: str, reason: str) -> dict:
    return {
        "has_text": True,
        "text_type": "other",
        "frame_text_sizes": ["none", "none", "none"],
        "frames_with_text": 0,
        "max_text_size": "none",
        "readable_text": False,
        "mirrored_text": False,
        "confidence": 0.0,
        "reason": reason[:500],
        "error": code,
    }


def _classify_action(decision: dict, min_remove_confidence: float = DEFAULT_MIN_REMOVE_CONFIDENCE) -> str:
    """Return keep_flip_ok, keep_no_flip, or remove using conservative rules."""
    if decision.get("error"):
        return "keep_no_flip"
    if not decision.get("has_text"):
        return "keep_flip_ok"

    text_type = str(decision.get("text_type") or "other")
    max_size = str(decision.get("max_text_size") or "none")
    size_rank = TEXT_SIZE_RANK.get(max_size, 0)
    frames_with_text = int(decision.get("frames_with_text") or 0)
    confidence = float(decision.get("confidence") or 0.0)

    # Natural writing is useful footage. Keep it, but never reverse it.
    if text_type in NATURAL_TEXT_TYPES:
        return "keep_no_flip"
    if confidence < min_remove_confidence:
        return "keep_no_flip"

    large = size_rank >= TEXT_SIZE_RANK["large"]
    medium_persistent = size_rank >= TEXT_SIZE_RANK["medium"] and frames_with_text >= 2

    if decision.get("mirrored_text") and size_rank >= TEXT_SIZE_RANK["medium"]:
        return "remove"
    if text_type in PROMINENT_TEXT_TYPES and large:
        return "remove"
    if text_type in {"subtitles", "title_card", "text_only_screen", "news_lower_third"} and medium_persistent:
        return "remove"
    if text_type in {"large_overlay", "ui_screen", "map_labels"} and medium_persistent:
        return "remove"
    return "keep_no_flip"


def _analyze_clip(path: str, model, processor, ratios: tuple[float, ...], max_new_tokens: int, scale_width: int) -> dict:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    frames = _extract_frames(path, ratios, scale_width)
    if len(frames) != len(ratios):
        for frame in frames:
            try:
                os.unlink(frame)
            except OSError:
                pass
        return _error_decision("frame_extraction", "Could not extract all requested frames; clip kept without mirroring.")

    try:
        images = []
        for frame in frames:
            with Image.open(frame) as img:
                images.append(img.convert("RGB"))

        content = [{"type": "image", "image": img} for img in images]
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
        return _normalize_decision(_parse_json_object(generated_text), generated_text, len(ratios))
    except Exception as exc:
        print(f"[cleanup] Analyze error for {os.path.basename(path)}: {exc}", flush=True)
        return _error_decision(str(type(exc).__name__), f"Analyze error; clip kept without mirroring: {exc}")
    finally:
        for frame in frames:
            try:
                os.unlink(frame)
            except OSError:
                pass


def _representative_order(clips: list[dict], pilot_size: int) -> tuple[list[dict], set[str]]:
    """Spread the pilot across the complete category-sorted index."""
    total = len(clips)
    pilot_size = min(max(0, pilot_size), total)
    if not pilot_size:
        return list(clips), set()
    if pilot_size == 1:
        indices = {0}
    else:
        indices = {round(i * (total - 1) / (pilot_size - 1)) for i in range(pilot_size)}
    pilot = [clips[i] for i in sorted(indices)]
    pilot_paths = {str(c.get("file") or "") for c in pilot}
    rest = [c for c in clips if str(c.get("file") or "") not in pilot_paths]
    return pilot + rest, pilot_paths


def _state_stats(records: list[dict]) -> dict:
    actions = Counter(str(r.get("action") or "") for r in records)
    errors = sum(1 for r in records if r.get("decision", {}).get("error"))
    total = len(records)
    return {
        "total": total,
        "actions": dict(actions),
        "remove_rate": actions.get("remove", 0) / total if total else 0.0,
        "error_rate": errors / total if total else 0.0,
        "errors": errors,
    }


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _guard_stats(stats: dict, max_delete_rate: float, max_error_rate: float, label: str) -> str:
    print(
        f"[cleanup] {label}: total={stats['total']} remove={stats['actions'].get('remove', 0)} "
        f"({stats['remove_rate']:.1%}) no_mirror={stats['actions'].get('keep_no_flip', 0)} "
        f"errors={stats['errors']} ({stats['error_rate']:.1%})",
        flush=True,
    )
    if stats["remove_rate"] > max_delete_rate:
        return f"remove rate {stats['remove_rate']:.1%} exceeds safety limit {max_delete_rate:.1%}"
    if stats["error_rate"] > max_error_rate:
        return f"error rate {stats['error_rate']:.1%} exceeds safety limit {max_error_rate:.1%}"
    return ""


def _resolve_library_root(clips: list[dict], configured_root: str) -> str:
    if configured_root:
        return os.path.abspath(configured_root)
    paths = [os.path.abspath(str(c.get("file"))) for c in clips if c.get("file")]
    if not paths:
        raise RuntimeError("Cannot determine library root from an empty index")
    common = os.path.commonpath(paths)
    if os.path.isfile(common) or os.path.splitext(common)[1]:
        common = os.path.dirname(common)
    return common


def _safe_relpath(path: str, root: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    if rel == ".." or rel.startswith(".." + os.sep) or os.path.isabs(rel):
        raise RuntimeError(f"Clip is outside library root: {path} (root {root})")
    return rel


def _text_safety_from_record(record: dict) -> dict:
    decision = record.get("decision", {})
    return {
        "version": SCRIPT_VERSION,
        "status": record.get("action", "keep_no_flip"),
        "has_visible_text": bool(decision.get("has_text")),
        "no_mirror": record.get("action") != "keep_flip_ok",
        "text_type": decision.get("text_type", "other"),
        "max_text_size": decision.get("max_text_size", "none"),
        "frames_with_text": int(decision.get("frames_with_text") or 0),
        "confidence": float(decision.get("confidence") or 0.0),
    }


def _apply_results(index_path: str, original_index: dict, state: dict, state_path: str, report_path: str,
                   index_dir: str, library_root: str, ratios: tuple[float, ...], args) -> None:
    backup_dir = os.path.join(index_dir, "backups")
    if not state.get("backup_index"):
        state["backup_index"] = _backup_index(index_path, backup_dir, "text_cleanup_v2")
    if not state.get("quarantine_dir"):
        state["quarantine_dir"] = os.path.join(index_dir, "quarantine_text_bad", state["run_id"])
    state["status"] = "applying"
    _atomic_write_json(state_path, state)

    remove_records = [r for r in state["done"].values() if r.get("action") == "remove"]
    print(f"[cleanup] Applying {len(remove_records)} quarantine moves...", flush=True)
    for number, record in enumerate(remove_records, start=1):
        src = os.path.abspath(record["file"])
        if not record.get("quarantine_path"):
            rel = _safe_relpath(src, library_root)
            record["quarantine_path"] = os.path.join(state["quarantine_dir"], rel)
            _atomic_write_json(state_path, state)
        dst = os.path.abspath(record["quarantine_path"])
        src_exists = os.path.exists(src)
        dst_exists = os.path.exists(dst)
        if src_exists and dst_exists:
            raise RuntimeError(f"Both source and quarantine file exist: {src} / {dst}")
        if not src_exists and not dst_exists:
            raise RuntimeError(f"Neither source nor quarantine file exists: {src} / {dst}")
        if src_exists:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        record["moved"] = True
        if number % max(1, args.save_every) == 0 or number == len(remove_records):
            _atomic_write_json(state_path, state)
            print(f"[cleanup] Quarantined {number}/{len(remove_records)}", flush=True)

    active_clips = []
    for clip in original_index.get("clips", []):
        path = str(clip.get("file") or "")
        record = state["done"].get(path)
        if not record:
            raise RuntimeError(f"Missing completed analysis record for {path}")
        if record.get("action") == "remove":
            continue
        annotated = dict(clip)
        annotated["text_safety"] = _text_safety_from_record(record)
        active_clips.append(annotated)

    stats = _state_stats(list(state["done"].values()))
    updated = dict(original_index)
    updated["clips"] = active_clips
    updated["total_dur"] = round(sum(float(c.get("duration", 3.0)) for c in active_clips), 2)
    updated["text_cleanup"] = {
        "version": SCRIPT_VERSION,
        "updated_at": _utc_stamp(),
        "frames": list(ratios),
        "min_remove_confidence": args.min_remove_confidence,
        "max_delete_rate": args.max_delete_rate,
        "max_error_rate": args.max_error_rate,
        "checked_total": stats["total"],
        "quarantined_total": stats["actions"].get("remove", 0),
        "no_mirror_total": stats["actions"].get("keep_no_flip", 0),
        "report_path": report_path,
        "state_path": state_path,
        "quarantine_dir": state["quarantine_dir"],
        "backup_index": state["backup_index"],
    }
    _atomic_write_json(index_path, updated)
    state["status"] = "complete"
    state["completed_at"] = _utc_stamp()
    _atomic_write_json(state_path, state)
    print(f"[cleanup] Index updated atomically: {index_path}", flush=True)


def _restore_previous_run(index_path: str, index_dir: str) -> int:
    old_state_path = os.path.join(index_dir, "text_cleanup_state.json")
    old_report_path = os.path.join(index_dir, "text_cleanup_report.jsonl")
    state = _load_json(old_state_path, {})
    if not state:
        print(f"[restore] No previous v1 state found: {old_state_path}", flush=True)
        return 0

    backup = str(state.get("backup_index") or "")
    quarantined = [q for q in state.get("quarantined", []) if isinstance(q, dict)]
    if not backup or not os.path.isfile(backup):
        raise RuntimeError(f"Previous index backup is missing: {backup}")
    restored_index = _load_json(backup, None)
    if not isinstance(restored_index, dict) or not isinstance(restored_index.get("clips"), list):
        raise RuntimeError(f"Invalid previous index backup: {backup}")

    conflicts = []
    missing = []
    for item in quarantined:
        raw_src = str(item.get("quarantine_path") or "")
        raw_dst = str(item.get("file") or "")
        if not raw_src or not raw_dst:
            missing.append((raw_src, raw_dst))
            continue
        src = os.path.abspath(raw_src)
        dst = os.path.abspath(raw_dst)
        src_exists = os.path.exists(src)
        dst_exists = os.path.exists(dst)
        if src_exists and dst_exists:
            conflicts.append((src, dst))
        elif not src_exists and not dst_exists:
            missing.append((src, dst))
    if conflicts or missing:
        raise RuntimeError(
            f"Restore preflight failed: conflicts={len(conflicts)}, missing={len(missing)}. Nothing was moved."
        )

    current_backup = _backup_index(index_path, os.path.join(index_dir, "backups"), "before_v1_restore")
    restored = 0
    already_present = 0
    for item in quarantined:
        src = os.path.abspath(str(item.get("quarantine_path") or ""))
        dst = os.path.abspath(str(item.get("file") or ""))
        if os.path.exists(dst):
            already_present += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        restored += 1

    _atomic_write_json(index_path, restored_index)

    state_archive = _archive_file(old_state_path, "restored_v1")
    report_archive = _archive_file(old_report_path, "restored_v1")
    print(f"[restore] Restored files: {restored}; already present: {already_present}", flush=True)
    print(f"[restore] Restored index: {index_path}", flush=True)
    print(f"[restore] Previous current-index backup: {current_backup}", flush=True)
    print(f"[restore] Archived state: {state_archive}", flush=True)
    if report_archive:
        print(f"[restore] Archived report: {report_archive}", flush=True)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Safely classify text in war clips and update index.json.")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--frames", default=",".join(str(x) for x in DEFAULT_FRAME_RATIOS))
    parser.add_argument("--pilot-clips", type=int, default=DEFAULT_PILOT_CLIPS)
    parser.add_argument("--max-delete-rate", type=float, default=DEFAULT_MAX_DELETE_RATE)
    parser.add_argument("--max-error-rate", type=float, default=DEFAULT_MAX_ERROR_RATE)
    parser.add_argument("--min-remove-confidence", type=float, default=DEFAULT_MIN_REMOVE_CONFIDENCE)
    parser.add_argument("--max-clips", type=int, default=0, help="Debug limit; disables applying results.")
    parser.add_argument("--force", action="store_true", help="Archive v2 state/report and analyze from scratch.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and classify without moving files or updating index.")
    parser.add_argument("--restore-previous-run", action="store_true", help="Restore files/index changed by the old v1 script.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--scale-width", type=int, default=896)
    parser.add_argument("--library-root", default="", help="Auto-detected from clip paths when omitted.")
    # Accepted only so an old idle watcher fails safely into the new policy.
    parser.add_argument("--min-confidence", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--delete-severities", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_clips > 0:
        args.dry_run = True
    index_path = os.path.abspath(args.index)
    index_dir = os.path.dirname(index_path)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")
    if args.restore_previous_run:
        return _restore_previous_run(index_path, index_dir)

    state_name = "text_cleanup_v2_dry_run_state.json" if args.dry_run else "text_cleanup_v2_state.json"
    report_name = "text_cleanup_v2_dry_run_report.jsonl" if args.dry_run else "text_cleanup_v2_report.jsonl"
    state_path = os.path.join(index_dir, state_name)
    report_path = os.path.join(index_dir, report_name)
    if args.force:
        existing_state = _load_json(state_path, {})
        if existing_state.get("status") == "applying":
            raise RuntimeError("Cannot force-restart while quarantine application is in progress; resume it first.")
        _archive_file(state_path, "forced_restart")
        _archive_file(report_path, "forced_restart")

    original_index = _load_json(index_path, {})
    clips = [c for c in original_index.get("clips", []) if isinstance(c, dict)]
    if args.max_clips > 0:
        clips = clips[:args.max_clips]
    if not clips:
        raise RuntimeError("Active index contains no clips")

    fingerprint = _index_fingerprint(clips)
    state = _load_json(state_path, {})
    if not state:
        state = {
            "version": SCRIPT_VERSION,
            "run_id": _utc_stamp(),
            "status": "analyzing",
            "started_at": _utc_stamp(),
            "index_fingerprint": fingerprint,
            "index_clip_count": len(clips),
            "done": {},
        }
        _atomic_write_json(state_path, state)
    if state.get("version") != SCRIPT_VERSION:
        raise RuntimeError(f"State belongs to another script version: {state.get('version')}")
    if state.get("status") == "complete":
        print("[cleanup] This run is already complete.", flush=True)
        return 0
    if (
        state.get("status") == "applying"
        and (state.get("index_fingerprint") != fingerprint or int(state.get("index_clip_count", -1)) != len(clips))
    ):
        source_backup = str(state.get("backup_index") or "")
        source_index = _load_json(source_backup, {})
        source_clips = [c for c in source_index.get("clips", []) if isinstance(c, dict)]
        if source_clips and _index_fingerprint(source_clips) == state.get("index_fingerprint"):
            original_index = source_index
            clips = source_clips
            fingerprint = _index_fingerprint(clips)
    if state.get("index_fingerprint") != fingerprint or int(state.get("index_clip_count", -1)) != len(clips):
        raise RuntimeError("Active index changed since this v2 run started. Use --force only after verifying the index.")

    ratios = _parse_frame_ratios(args.frames)
    library_root = _resolve_library_root(clips, args.library_root)
    ordered, pilot_paths = _representative_order(clips, args.pilot_clips)
    done = state.setdefault("done", {})

    print(f"[cleanup] Version: {SCRIPT_VERSION}", flush=True)
    print(f"[cleanup] Index: {index_path}", flush=True)
    print(f"[cleanup] Clips: {len(clips)}; already analyzed: {len(done)}", flush=True)
    print(f"[cleanup] Frames: {ratios}; library root: {library_root}", flush=True)
    print(f"[cleanup] Pilot: {len(pilot_paths)}; maximum delete rate: {args.max_delete_rate:.0%}", flush=True)
    print("[cleanup] Analysis phase does not move files or modify index.json.", flush=True)

    pilot_records = [done[p] for p in pilot_paths if p in done]
    if len(pilot_records) == len(pilot_paths) and pilot_paths:
        problem = _guard_stats(_state_stats(pilot_records), args.max_delete_rate, args.max_error_rate, "Pilot resume check")
        if problem:
            state["status"] = "pilot_blocked"
            state["blocked_reason"] = problem
            _atomic_write_json(state_path, state)
            print(f"[cleanup] SAFETY STOP: {problem}. No files were moved.", flush=True)
            return 2
        state["pilot_passed"] = True

    remaining = [c for c in ordered if str(c.get("file") or "") not in done]
    if remaining:
        model, processor = _load_model(args.model_path)
        start = time.time()
        checked_now = 0
        for clip in remaining:
            path = str(clip.get("file") or "")
            if not path or not os.path.exists(path):
                decision = _error_decision("missing_file", "File missing before analysis; kept without mirroring.")
            else:
                decision = _analyze_clip(
                    path,
                    model,
                    processor,
                    ratios=ratios,
                    max_new_tokens=args.max_new_tokens,
                    scale_width=args.scale_width,
                )
            action = _classify_action(decision, args.min_remove_confidence)
            record = {
                "checked_at": _utc_stamp(),
                "clip_id": clip.get("id") or os.path.splitext(os.path.basename(path))[0],
                "file": path,
                "action": action,
                "decision": decision,
            }
            done[path] = record
            _append_jsonl(report_path, record)
            checked_now += 1

            pilot_records = [done[p] for p in pilot_paths if p in done]
            if pilot_paths and not state.get("pilot_passed") and len(pilot_records) == len(pilot_paths):
                problem = _guard_stats(
                    _state_stats(pilot_records), args.max_delete_rate, args.max_error_rate, "Representative pilot"
                )
                if problem:
                    state["status"] = "pilot_blocked"
                    state["blocked_reason"] = problem
                    _atomic_write_json(state_path, state)
                    print(f"[cleanup] SAFETY STOP: {problem}. No files were moved.", flush=True)
                    return 2
                state["pilot_passed"] = True
                print("[cleanup] Pilot passed. Continuing with the complete library.", flush=True)

            if checked_now % max(1, args.save_every) == 0 or len(done) == len(clips):
                _atomic_write_json(state_path, state)
                elapsed = max(0.001, time.time() - start)
                rate = checked_now / elapsed
                eta = (len(clips) - len(done)) / rate if rate > 0 else 0.0
                stats = _state_stats(list(done.values()))
                print(
                    f"[cleanup] {len(done)}/{len(clips)} analyzed | remove={stats['actions'].get('remove', 0)} "
                    f"| no_mirror={stats['actions'].get('keep_no_flip', 0)} | errors={stats['errors']} "
                    f"| {rate:.3f} clips/s | ETA {eta/3600:.1f}h",
                    flush=True,
                )
                _empty_cuda_cache()

    if len(done) != len(clips):
        raise RuntimeError(f"Analysis incomplete: {len(done)}/{len(clips)}")

    final_stats = _state_stats(list(done.values()))
    problem = _guard_stats(final_stats, args.max_delete_rate, args.max_error_rate, "Full-library safety check")
    if problem:
        state["status"] = "final_blocked"
        state["blocked_reason"] = problem
        _atomic_write_json(state_path, state)
        print(f"[cleanup] SAFETY STOP: {problem}. No files were moved and index.json was not changed.", flush=True)
        return 2

    state["status"] = "analyzed"
    state["final_stats"] = final_stats
    _atomic_write_json(state_path, state)
    if args.dry_run:
        print("[cleanup] Dry run complete. No files moved; index.json unchanged.", flush=True)
        return 0

    _apply_results(index_path, original_index, state, state_path, report_path, index_dir, library_root, ratios, args)
    print("[cleanup] DONE", flush=True)
    print(f"[cleanup] Kept with flip: {final_stats['actions'].get('keep_flip_ok', 0)}", flush=True)
    print(f"[cleanup] Kept without flip: {final_stats['actions'].get('keep_no_flip', 0)}", flush=True)
    print(f"[cleanup] Quarantined: {final_stats['actions'].get('remove', 0)}", flush=True)
    print(f"[cleanup] Report: {report_path}", flush=True)
    print(f"[cleanup] State: {state_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
