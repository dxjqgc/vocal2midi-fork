import pathlib

from inference.device_utils import resolve_onnx_providers
from inference.HubertFA.onnx_infer import InferenceOnnx


_HFA_REPAIR_IGNORE_TOKENS = {"SP", "AP", "EP", "br", "sil", "pau"}
_HFA_REPAIR_RATIO = 0.6
_HFA_REPAIR_MIN_GLOBAL_K = 3
_HFA_REPAIR_GLOBAL_PORTION = 0.15

# Crushed-syllable run redistribution (root-cause fix for leak-noise
# absorption): a run of >= 2 consecutive lexical words crushed below
# _HFA_CRUSHED_MAX_SEC whose preceding anchor is inflated gets the span
# [anchor.start, next normal word start] evenly redistributed.
_HFA_CRUSHED_MAX_SEC = 0.06
_HFA_CRUSHED_MIN_RUN = 2
_HFA_REDISTRIBUTE_MIN_SHARE_SEC = 0.07
_HFA_REDISTRIBUTE_ANCHOR_RATIO = 2.0
# False-positive guard: every run member must be MUCH shorter than the even
# share it would receive. Genuine fast syllables (e.g. staccato 是) sit near
# the even share already; leak-noise-crushed words are 5-10x below it.
_HFA_CRUSHED_RECOVERY_RATIO = 0.5


def _is_lexical_word(word) -> bool:
    return getattr(word, "text", None) not in _HFA_REPAIR_IGNORE_TOKENS


def _word_duration(word) -> float:
    return max(0.0, word.end - word.start)


def _rescale_word_times(word, new_start: float, new_end: float) -> None:
    """Remap word and phoneme boundaries linearly onto [new_start, new_end]."""
    old_start = word.start
    old_span = max(1e-9, word.end - old_start)
    scale = (new_end - new_start) / old_span
    word.start = new_start
    word.end = new_end
    for phoneme in word.phonemes:
        p_start = new_start + (phoneme.start - old_start) * scale
        p_end = new_start + (phoneme.end - old_start) * scale
        phoneme.start = min(max(p_start, new_start), new_end)
        phoneme.end = min(max(p_end, new_start), new_end)


def _repair_short_word_boundaries(words) -> list[str]:
    """
    Repair abnormally short lexical words produced by HFA.

    Rule of thumb:
    1. the current lexical word must be among the globally shortest few;
    2. it must also be locally much shorter than both surrounding lexical words;
    3. there must be a later lexical word, so the current word is not the last note.

    When triggered, the current word end is stretched to the next lexical word start,
    crossing over silence tags such as SP if necessary.
    """
    lexical_indices = [idx for idx, word in enumerate(words) if _is_lexical_word(word)]
    if len(lexical_indices) < 3:
        return []

    lexical_durations = [(idx, max(0.0, words[idx].end - words[idx].start)) for idx in lexical_indices]
    sorted_by_duration = sorted(lexical_durations, key=lambda item: item[1])
    shortest_k = max(_HFA_REPAIR_MIN_GLOBAL_K, int(len(lexical_indices) * _HFA_REPAIR_GLOBAL_PORTION + 0.999999))
    shortest_candidates = {idx for idx, _ in sorted_by_duration[:shortest_k]}

    repair_logs = []
    remove_indices = set()
    for pos in range(1, len(lexical_indices) - 1):
        cur_idx = lexical_indices[pos]
        if cur_idx in remove_indices or cur_idx not in shortest_candidates:
            continue

        prev_idx = lexical_indices[pos - 1]
        next_idx = lexical_indices[pos + 1]
        cur_word = words[cur_idx]
        prev_word = words[prev_idx]
        next_word = words[next_idx]
        middle_words = words[cur_idx + 1:next_idx]

        cur_dur = max(0.0, cur_word.end - cur_word.start)
        prev_dur = max(0.0, prev_word.end - prev_word.start)
        next_dur = max(0.0, next_word.end - next_word.start)
        neighbor_min = min(prev_dur, next_dur)

        if neighbor_min <= 0:
            continue
        if cur_dur >= neighbor_min * _HFA_REPAIR_RATIO:
            continue
        if next_word.start <= cur_word.end:
            continue
        if any(_is_lexical_word(word) for word in middle_words):
            continue

        old_end = cur_word.end
        cur_word.move_end(next_word.start)
        removed_texts = [word.text for word in middle_words]
        remove_indices.update(range(cur_idx + 1, next_idx))
        repair_logs.append(
            f"[HFA Repair] '{cur_word.text}' {cur_word.start:.4f}-{old_end:.4f} -> {cur_word.start:.4f}-{cur_word.end:.4f}"
            + (f"; removed fillers={removed_texts}" if removed_texts else "")
        )

    if remove_indices:
        kept_words = [word for idx, word in enumerate(words) if idx not in remove_indices]
        words.clear()
        words.extend(kept_words)

    return repair_logs


def _redistribute_crushed_runs(words) -> list[str]:
    """Redistribute anchor + crushed-syllable runs (leak-noise absorption).

    When residual accompaniment noise precedes the first sung character, HFA
    absorbs it into the leading word's initial consonant (e.g. /w/ stretched
    to ~1.1s) and time-conservation crushes the following back-to-back words
    to tens of milliseconds. The existing single-word repair cannot fire
    here: its `next_word.start > cur_word.end` guard requires a gap, but
    crushed words are seamless by construction.

    Detection mirrors the GuitarSheetGenerator backend's proven repair
    (v2m_melody_service Pass 1): a run of >= 2 consecutive lexical words
    shorter than _HFA_CRUSHED_MAX_SEC whose preceding anchor is inflated
    (>= _HFA_REDISTRIBUTE_ANCHOR_RATIO x even share) gets the span
    [anchor.start, first normal word after the run] evenly split between
    anchor + run members. HFA word boundaries are seamless, so the span is
    time-conserving and an even split falls within one sixteenth note of the
    true rhythm.
    """
    lexical_indices = [idx for idx, word in enumerate(words) if _is_lexical_word(word)]
    if len(lexical_indices) < 3:
        return []

    repair_logs = []
    pos = 0
    while pos < len(lexical_indices):
        idx = lexical_indices[pos]
        if _word_duration(words[idx]) >= _HFA_CRUSHED_MAX_SEC:
            pos += 1
            continue

        # grow a run of consecutive crushed lexical words (seamless by
        # construction, but tolerate up to 1ms of float drift)
        run_end = pos
        while run_end + 1 < len(lexical_indices):
            nxt_idx = lexical_indices[run_end + 1]
            cur_word = words[lexical_indices[run_end]]
            nxt_word = words[nxt_idx]
            if (_word_duration(nxt_word) >= _HFA_CRUSHED_MAX_SEC
                    or nxt_word.start - cur_word.end > 0.001):
                break
            run_end += 1
        if run_end - pos + 1 < _HFA_CRUSHED_MIN_RUN:
            pos = run_end + 1
            continue

        run_indices = lexical_indices[pos:run_end + 1]
        if pos == 0:
            pos = run_end + 1
            continue  # no anchor before the run
        anchor_idx = lexical_indices[pos - 1]
        anchor = words[anchor_idx]

        after_idx = lexical_indices[run_end + 1] if run_end + 1 < len(lexical_indices) else None
        span_start = anchor.start
        span_end = words[after_idx].start if after_idx is not None else words[run_indices[-1]].end

        members = [anchor_idx] + run_indices
        share = (span_end - span_start) / len(members)
        if share < _HFA_REDISTRIBUTE_MIN_SHARE_SEC:
            pos = run_end + 1
            continue  # span itself is tiny — not an inflated-anchor case
        if _word_duration(anchor) < share * _HFA_REDISTRIBUTE_ANCHOR_RATIO:
            pos = run_end + 1
            continue  # anchor not inflated — shorts may be genuine fast syllables
        if any(_word_duration(words[i]) >= share * _HFA_CRUSHED_RECOVERY_RATIO
               for i in run_indices):
            pos = run_end + 1
            continue  # a run member already holds a fair share — genuine fast
                      # syllables, not leak-noise crushing

        cursor = span_start
        for member_idx in members:
            word = words[member_idx]
            _rescale_word_times(word, cursor, cursor + share)
            cursor += share

        crushed_texts = [words[i].text for i in run_indices]
        repair_logs.append(
            f"[HFA Repair] crushed-run redistributed: anchor '{anchor.text}' "
            f"(dur {_word_duration(anchor):.3f}s) + {crushed_texts} -> "
            f"{len(members)} x {share:.3f}s over [{span_start:.3f}s, {span_end:.3f}s]"
        )
        pos = run_end + 1

    return repair_logs


def _repair_pred_dict_short_words(pred_dict) -> None:
    total_repaired = 0
    for stem, pred in pred_dict.items():
        if len(pred) < 3:
            continue
        words = pred[2]
        repair_logs = _repair_short_word_boundaries(words)
        # run the root-cause redistribution after the single-word repair so
        # genuine short words separated by silence are already fixed and the
        # crushed-run detector sees a cleaner sequence
        repair_logs += _redistribute_crushed_runs(words)
        if repair_logs:
            print(f"[HFA Repair] {stem}: repaired {len(repair_logs)} short word(s)")
            for log in repair_logs:
                print(log)
            total_repaired += len(repair_logs)

    if total_repaired > 0:
        print(f"[HFA Repair] Total repaired short words: {total_repaired}")


def load_hfa_model(model_dir, device="dml"):
    """
    Load the HubertFA ONNX model on DirectML by default, with CPU fallback.
    """
    print("Loading HubertFA ONNX model...")
    model = InferenceOnnx(onnx_path=pathlib.Path(model_dir) / 'model.onnx')
    model.load_config()
    model.init_decoder()
    import onnxruntime as ort
    provider_name, providers = resolve_onnx_providers(device, label="HubertFA ONNX")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    model.model = ort.InferenceSession(str(model.model_folder / 'model.onnx'), options, providers=providers)
    print(f"HubertFA ONNX session created with provider={provider_name}: {model.model.get_providers()}")
    return model

def run_hubert_fa(hfa_model, temp_dir, language="zh", cancel_checker=None, use_phoneme_g2p=False):
    print("[Hybrid Pipeline] Running HubertFA forced alignment...")
    if cancel_checker and cancel_checker():
        raise InterruptedError("HFA 任务已取消")
    hfa_model.dataset = []
    hfa_model.predictions = []
    
    dict_file = "ds-zh-pinyin-lite.txt" if language == "zh" else "japanese_dict_full.txt"
    dict_path = hfa_model.vocab_folder / dict_file

    if use_phoneme_g2p:
        # The input .lab file already contains phoneme tokens.
        # For Japanese, add an extra "phoneme -> mora word boundary" pass so HFA
        # still aligns at the phoneme level while word boundaries better match mora units.
        g2p_mode = "ja_mora_phoneme" if language == "ja" else "phoneme"
        hfa_model.get_dataset(wav_folder=temp_dir, language=language, g2p=g2p_mode, dictionary_path=None)
    else:
        hfa_model.get_dataset(wav_folder=temp_dir, language=language, g2p="dictionary", dictionary_path=dict_path)
    if cancel_checker and cancel_checker():
        raise InterruptedError("HFA 任务已取消")
    if len(hfa_model.dataset) > 0:
        nl_phonemes = "AP" if language == "zh" else ""
        hfa_model.infer(non_lexical_phonemes=nl_phonemes, pad_times=1, pad_length=5)

    pred_dict = {p[0].stem: p for p in hfa_model.predictions}
    _repair_pred_dict_short_words(pred_dict)
    return pred_dict

def export_hfa_artifacts(chunks, temp_dir_path, hfa_model, output_key, output_dir, output_formats, cancel_checker=None):
    import shutil

    output_formats = set(output_formats or [])
    export_chunks = "chunks" in output_formats
                                           
    export_textgrid = ("textgrid" in output_formats) or export_chunks

    tg_subfolder = None
    if export_textgrid:
        if cancel_checker and cancel_checker():
            raise InterruptedError("HFA 导出任务已取消")
        temp_tg_dir = temp_dir_path / "temp_tg"
        hfa_model.export(temp_tg_dir, output_format=['textgrid'])
        tg_subfolder = temp_tg_dir / "TextGrid"
    
    for chunk_idx, chunk in enumerate(chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("HFA 导出任务已取消")
        stem = f"chunk_{chunk_idx}"
        new_stem = f"{output_key}_{chunk_idx:03d}"
        
        if export_chunks:
            chunk_wav_path = temp_dir_path / f"{stem}.wav"
            try:
                if chunk_wav_path.exists():
                    shutil.copy2(chunk_wav_path, output_dir / f"{new_stem}.wav")
                else:
                    print(f"[Warning] Chunk WAV file not found, skipping: {chunk_wav_path}")
            except Exception as e:
                print(f"[Error] Failed to copy chunk {chunk_wav_path}: {e}")

        if tg_subfolder is not None and tg_subfolder.exists():
            tg_path = tg_subfolder / f"{stem}.TextGrid"
            try:
                if tg_path.exists():
                    shutil.copy2(tg_path, output_dir / f"{new_stem}.TextGrid")
            except Exception as e:
                print(f"[Error] Failed to copy TextGrid {tg_path}: {e}")
