"""
War niche clip selector — picks clips by semantic similarity (Vertex embeddings).
Falls back to category-random if embeddings unavailable.
"""

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backend import api_client

CATEGORIES = ["infantry", "armor", "artillery", "drones", "aviation", "naval", "explosions"]

LIBRARY_PATH = "/workspace/gdrive/library/russia_ukraine_war"
INDEX_PATH = "/workspace/FAA/movies/russia_ukraine_war/index.json"

SEMANTIC_MIN_SIM = 0.20

_CATEGORY_PROMPT = """You are a video editor for a war documentary channel. Given a script segment, decide which visual category best matches it.

Categories:
- infantry: soldiers, ground troops, combat footage, positions, trenches
- armor: tanks, APCs, armored vehicles, convoys
- artillery: howitzers, MLRS, shelling, HIMARS, rocket launchers
- drones: FPV drones, reconnaissance drones, drone strikes, UAVs
- aviation: jets, helicopters, air strikes, aircraft
- naval: ships, boats, sea operations, naval strikes
- explosions: big explosions, blasts, ammunition depots, impacts

Reply with ONLY the category name. If unclear, reply "explosions".

Segment: {text}"""

_ENTITY_PROMPT = """Extract named entities from this war video script segment that should be shown as on-screen labels. Only extract:
- City/location names (Bakhmut, Kherson, Avdiivka, etc.)
- Specific weapon/vehicle names (Leopard 2, HIMARS, Bayraktar, etc.)
- Military unit names if mentioned

Reply as JSON array of strings. If nothing specific, reply [].
Max 2 entities per segment.

Segment: {text}"""

# Cached index data
_INDEX_CACHE = None


def _load_index():
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    if not os.path.exists(INDEX_PATH):
        _INDEX_CACHE = []
        return []
    with open(INDEX_PATH) as f:
        data = json.load(f)
    clips = data.get("clips", [])
    _INDEX_CACHE = clips
    return clips


def _has_embeddings(clips):
    if not clips:
        return False
    with_emb = sum(1 for c in clips if c.get("embedding"))
    return with_emb >= len(clips) * 0.8


def _call_rewrite_api(prompt):
    try:
        text, _ = api_client.call_rewrite_api(
            "Reply with exactly the requested format. No markdown.",
            [{"role": "user", "content": prompt}],
            timeout=30,
            max_retries=2,
            step_label="war_clip_selector",
        )
        return text.strip()
    except Exception as e:
        print(f"[war_clip_selector] A6API call failed: {e}", flush=True)
    return None


_LOCAL_LLM = None
_LOCAL_PROC = None


def _ensure_local_llm():
    global _LOCAL_LLM, _LOCAL_PROC
    if _LOCAL_LLM is not None:
        return _LOCAL_LLM, _LOCAL_PROC
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_path = "/workspace/models/qwen2.5-vl-3b"
    print("[war_clip_selector] Loading Qwen2.5-VL-3B for text classification...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    _LOCAL_LLM = model
    _LOCAL_PROC = processor
    print("[war_clip_selector] LLM ready.")
    return model, processor


def _llm_classify_text(text):
    import torch
    model, processor = _ensure_local_llm()
    prompt = "Classify this war video script segment into ONE category: infantry, armor, artillery, drones, aviation, naval, explosions. Segment: " + text + ". Category:"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    tok_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[tok_text], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True).strip().lower().replace(" ", "_")
    for cat in CATEGORIES:
        if cat in response:
            return cat
    return "explosions"


def classify_segment(text):
    prompt = _CATEGORY_PROMPT.format(text=text)
    result = _call_rewrite_api(prompt)
    if result:
        result = result.lower().strip().replace(" ", "_")
        for cat in CATEGORIES:
            if cat in result:
                return cat
    return _llm_classify_text(text)


def extract_entities(text):
    prompt = _ENTITY_PROMPT.format(text=text)
    result = _call_rewrite_api(prompt)
    if result:
        try:
            entities = json.loads(result)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities[:2] if e]
        except json.JSONDecodeError:
            matches = re.findall(r'"([^"]+)"', result)
            return matches[:2]
    return []


def _get_clips_from_category(category):
    cat_path = os.path.join(LIBRARY_PATH, category)
    if not os.path.isdir(cat_path):
        return []
    return [os.path.join(cat_path, f) for f in os.listdir(cat_path)
            if f.endswith((".mp4", ".mkv", ".webm"))]


def select_clips_for_war(segments, audio_dur, global_used_ids=None, emit=None):
    """
    Select clips for war niche segments.
    Strategy 1 (preferred): Semantic — embed segment text, cosine match against clip embeddings.
    Strategy 2 (fallback): Category — classify segment, pick random from that category.
    """
    used_files = set(global_used_ids or [])
    clip_data = []

    all_clips = _load_index()
    use_semantic = _has_embeddings(all_clips)

    if use_semantic:
        import numpy as np
        from backend.embeddings import embed_texts
        if emit:
            emit("clips", f"War selector: semantic mode ({len(all_clips)} indexed clips)")

        # Batch embed ALL segment texts in one Vertex API call (~2 sec)
        seg_texts = [seg.get("text", "") or "war footage" for seg in segments]
        if emit:
            emit("clips", f"Embedding {len(seg_texts)} segments (batch)...")
        seg_vectors = embed_texts(seg_texts, emit=lambda t, m: emit("clips", m) if emit else None)

        if not seg_vectors:
            use_semantic = False
        else:
            # Pre-filter clips that exist on disk and have embeddings
            valid_clips = [c for c in all_clips if c.get("embedding") and os.path.exists(c.get("file", ""))]
            if emit:
                emit("clips", f"Matching against {len(valid_clips)} clips...")

            # Build numpy matrices for fast cosine similarity
            clip_matrix = np.array([c["embedding"] for c in valid_clips], dtype=np.float32)
            seg_matrix = np.array(seg_vectors, dtype=np.float32)

            # Normalize for cosine similarity
            clip_norms = np.linalg.norm(clip_matrix, axis=1, keepdims=True)
            clip_norms[clip_norms == 0] = 1.0
            clip_matrix_norm = clip_matrix / clip_norms

            seg_norms = np.linalg.norm(seg_matrix, axis=1, keepdims=True)
            seg_norms[seg_norms == 0] = 1.0
            seg_matrix_norm = seg_matrix / seg_norms

            # All similarities at once: (num_segments x num_clips)
            sim_matrix = seg_matrix_norm @ clip_matrix_norm.T

            if emit:
                emit("clips", "Similarity matrix computed, picking clips...")

            # Build category index for each clip
            clip_categories = [c.get("scene_type", "") for c in valid_clips]

            for i, seg in enumerate(segments):
                seg_dur = seg.get("end", 0) - seg.get("start", 0)
                if seg_dur <= 0:
                    seg_dur = 3.0

                sims = sim_matrix[i]

                # Mask already used clips
                mask = np.ones(len(valid_clips), dtype=bool)
                for idx, c in enumerate(valid_clips):
                    if c.get("file") in used_files:
                        mask[idx] = False

                masked_sims = sims.copy()
                masked_sims[~mask] = -1.0

                # Pick best overall
                best_idx = int(np.argmax(masked_sims))
                best_sim = masked_sims[best_idx]

                if best_sim >= SEMANTIC_MIN_SIM:
                    pick = valid_clips[best_idx]
                    used_files.add(pick["file"])
                    clip_data.append({"file": pick["file"], "duration": seg_dur})
                else:
                    # Fallback: any unused clip
                    for idx in range(len(valid_clips)):
                        if mask[idx]:
                            pick = valid_clips[idx]
                            used_files.add(pick["file"])
                            clip_data.append({"file": pick["file"], "duration": seg_dur})
                            break

                if emit and (i + 1) % 50 == 0:
                    emit("clips", f"War clips: {i+1}/{len(segments)} matched")
    else:
        category_clips = {cat: _get_clips_from_category(cat) for cat in CATEGORIES}
        total_available = sum(len(v) for v in category_clips.values())
        if emit:
            emit("clips", f"War selector: category mode ({total_available} clips, no embeddings)")

        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            seg_dur = seg.get("end", 0) - seg.get("start", 0)
            if seg_dur <= 0:
                seg_dur = 3.0
            category = classify_segment(text)
            available = [c for c in category_clips[category] if c not in used_files]
            if not available:
                available = category_clips[category]
            if not available:
                available = category_clips.get("explosions", [])
            if available:
                clip_file = random.choice(available)
                used_files.add(clip_file)
                clip_data.append({"file": clip_file, "duration": seg_dur})

            if emit and (i + 1) % 10 == 0:
                emit("clips", f"War clips: {i+1}/{len(segments)} (cat: {category})")

    clip_data = [c for c in clip_data if c is not None]
    if emit:
        emit("clips", f"Selected {len(clip_data)} war clips.")
    return clip_data


def generate_war_badges(segments):
    badges = []
    for i in range(0, len(segments), 3):
        seg = segments[i]
        text = seg.get("text", "")
        start = seg.get("start", 0)
        end = seg.get("end", start + 3)
        entities = extract_entities(text)
        for ent in entities:
            if len(ent) > 2 and len(ent) < 25:
                badges.append({
                    "text": ent,
                    "start": start + 0.3,
                    "end": min(end, start + 3.0),
                })
    return badges


def badges_to_overlays(badges):
    BADGE_COLORS = [
        {"bg": "0xE63946@0.9", "fg": "white"},
        {"bg": "0xF77F00@0.9", "fg": "white"},
        {"bg": "0x2196F3@0.9", "fg": "white"},
        {"bg": "0x4CAF50@0.9", "fg": "white"},
        {"bg": "0xFFD600@0.9", "fg": "black"},
    ]
    overlays = []
    for i, badge in enumerate(badges):
        color_set = BADGE_COLORS[i % len(BADGE_COLORS)]
        overlays.append({
            "text": badge["text"].upper(),
            "start": badge["start"],
            "duration": badge["end"] - badge["start"],
            "position": "bottom-left",
            "size": 28,
            "color": color_set["fg"],
            "bg_color": color_set["bg"],
        })
    return overlays
