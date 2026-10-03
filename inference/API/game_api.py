import pathlib
import sys
import traceback

import librosa
import numpy as np

# Add repo path if needed
ROOT_DIR = pathlib.Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.io.note_io import NoteInfo, pad_1d_arrays
from inference.game.alignment_utils import align_notes_to_words  # noqa: F401 (kept for external callers)
from inference.game.onnx_runtime import GameOnnxModel


_SINGABLE_JA_PHONEMES = {"a", "i", "u", "e", "o"}
_SINGABLE_ZH_FALLBACK_VOWELS = set("aeiouv")
_NON_SINGABLE_WORD_TOKENS = {"SP", "AP", "EP", "br", "sil", "pau"}

# Lyric-to-note DP alignment costs (seconds of onset-distance equivalent).
# Skip-word must cost more than any in-window offset match (up to
# _DP_MAX_MATCH_DIST): gluing a character onto a neighbor's note is a
# lyric-integrity violation, reserved for words with no note to match at all.
_DP_MAX_MATCH_DIST = 0.6
_DP_SKIP_WORD_COST = 0.75
_DP_REST_PROMOTE_PENALTY = 0.15


def _normalize_ts(ts) -> list[float]:
    if ts is None:
        return []
    if hasattr(ts, "detach"):
        ts = ts.detach().cpu().tolist()
    elif hasattr(ts, "tolist"):
        ts = ts.tolist()
    return [float(t) for t in ts]


def _resolve_game_language_id(game_model: GameOnnxModel, language: str | None = None) -> int:
    del game_model, language
    return 0


def _normalize_phone_text(phone_text: str) -> str:
    return str(phone_text or "").split("/")[-1].strip()


def _is_singable_phone(phone_text: str, language: str | None) -> bool:
    phone_raw = _normalize_phone_text(phone_text)
    phone = phone_raw.lower()
    lang = (language or "").lower()

    if not phone or phone == "sp":
        return False
    if lang == "ja":
        return phone in _SINGABLE_JA_PHONEMES or phone_raw == "N"

    if phone in {"n", "ng", "m"}:
        return False
    return any(ch in _SINGABLE_ZH_FALLBACK_VOWELS for ch in phone)


def _find_word_nucleus_start(word, language: str | None) -> float | None:
    phonemes = getattr(word, "phonemes", None) or []
    for phoneme in phonemes:
        if _is_singable_phone(getattr(phoneme, "text", ""), language):
            return float(phoneme.start)
    return None


def load_game_model(model_dir: str, device="dml"):
    """
    Loads the GAME ONNX model suite.
    """
    print(f"Loading GAME ONNX model from '{model_dir}'...")
    try:
        model = GameOnnxModel(pathlib.Path(model_dir), requested_device=device)
    except Exception as e:
        raise RuntimeError(
            f"Error loading GAME ONNX model: {e}\n"
            "Please ensure the GAME ONNX model directory and its contents are correct."
        )

    print(f"GAME ONNX model loaded successfully with provider: {model.provider_name}.")
    return model


def extract_vowel_boundaries(result_word, original_chars: list[str], language: str | None = None):
    word_durs = []
    word_vuvs = []
    lyrics = []

    char_idx = 0
    last_end = 0.0

    ignore_tokens = _NON_SINGABLE_WORD_TOKENS
    is_romaji = len(original_chars) > 0 and all(c.isascii() or c == "" for c in original_chars)

    for i, word in enumerate(result_word):
        if word.text in ignore_tokens:
            if word.end > last_end:
                word_durs.append(word.end - last_end)
                word_vuvs.append(0)
                lyrics.append("")
                last_end = word.end
            continue

        vowel_start = _find_word_nucleus_start(word, language)
        if vowel_start is None:
            if word.end > last_end:
                word_durs.append(word.end - last_end)
                word_vuvs.append(0)
                lyrics.append("")
                last_end = word.end
            continue

        if vowel_start > last_end + 0.005:
            word_durs.append(vowel_start - last_end)
            word_vuvs.append(0)
            lyrics.append("")
        elif vowel_start < last_end:
            vowel_start = last_end

        next_vowel_start = word.end
        if i + 1 < len(result_word):
            next_w = result_word[i + 1]
            if next_w.text not in ignore_tokens:
                next_nucleus_start = _find_word_nucleus_start(next_w, language)
                if next_nucleus_start is not None:
                    next_vowel_start = next_nucleus_start

        note_end = next_vowel_start
        if i + 1 < len(result_word) and result_word[i + 1].text in ignore_tokens:
            note_end = word.end

        dur = note_end - vowel_start
        if dur < 0:
            dur = 0.0

        word_durs.append(dur)
        word_vuvs.append(1)

        if is_romaji:
            while char_idx < len(original_chars) and original_chars[char_idx].lower() != word.text.lower():
                char_idx += 1
            if char_idx < len(original_chars):
                lyrics.append(original_chars[char_idx])
                char_idx += 1
            else:
                lyrics.append(word.text)
        else:
            if char_idx < len(original_chars):
                lyrics.append(original_chars[char_idx])
                char_idx += 1
            else:
                lyrics.append(word.text)

        last_end = note_end

    return word_durs, word_vuvs, lyrics


def _run_game_inference_batch(
    *,
    game_model: GameOnnxModel,
    waveforms_np: list[np.ndarray],
    waveform_durations_np: list[float],
    known_durations_np: list[np.ndarray] | None,
    seg_threshold: float,
    seg_radius: float,
    est_threshold: float,
    ts,
    language: str | None = None,
):
    padded_wavs = pad_1d_arrays(waveforms_np).astype(np.float32, copy=False)
    padded_kd = None
    if known_durations_np is not None:
        padded_kd = pad_1d_arrays(known_durations_np, pad_value=0.0).astype(np.float32, copy=False)

    boundary_radius = int(round(seg_radius / game_model.timestep))
    return game_model.infer_batch(
        waveforms=padded_wavs,
        durations=np.asarray(waveform_durations_np, dtype=np.float32),
        known_durations=padded_kd,
        boundary_threshold=float(seg_threshold),
        boundary_radius=boundary_radius,
        score_threshold=float(est_threshold),
        language=_resolve_game_language_id(game_model, language),
        ts=_normalize_ts(ts),
    )


def _assign_lyrics_dp(word_durs, word_vuvs, lyrics, note_dur, note_presence):
    """Monotonic DP matching of lyric characters onto GAME notes.

    The word grid from HFA (even after repair) can disagree with GAME's
    audio-driven segmentation by several hundred ms (observed: chunk where
    HFA crushed three syllables to 3x0.157s while GAME heard 0.54/0.20/0.17).
    Cropping GAME notes to the word grid destroys the real rhythm, so instead
    the GAME note timeline passes through untouched and characters are
    attached by onset proximity under a monotonic (Needleman-Wunsch style)
    alignment: neighbors constrain each other, so local grid skew cannot
    cascade into a global drift.

    A character may land on a rest note (presence=0), promoting it to a
    sounding note with GAME's pitch estimate, at an extra cost — this
    recovers words GAME judged unvoiced that are really sung (observed: 是
    lost entirely under the old grid-cropping loop). Words that match no
    note carry their character onto the next matched note's lyric, so no
    character is ever silently dropped.

    Returns (assigned, n_promoted, n_carried): per-note lyric — the matched
    character, "-" for unmatched sounding notes, "" for rests.
    """
    word_onsets = np.cumsum([0.0] + list(word_durs))[:-1] if word_durs else []
    words = [
        (float(word_onsets[i]), str(lyrics[i]))
        for i in range(min(len(lyrics), len(word_vuvs), len(word_onsets)))
        if word_vuvs[i] == 1 and lyrics[i]
    ]
    note_onsets = np.cumsum([0.0] + list(note_dur))[:-1]
    assigned = ["-" if p else "" for p in note_presence]
    if not words or len(note_dur) == 0:
        return assigned, 0, 0

    n_words, n_notes = len(words), len(note_dur)
    inf = float("inf")
    # cost[wi][ni]: min cost after consuming wi words / ni notes
    cost = [[inf] * (n_notes + 1) for _ in range(n_words + 1)]
    back: list = [[None] * (n_notes + 1) for _ in range(n_words + 1)]
    cost[0][0] = 0.0
    for wi in range(n_words + 1):
        for ni in range(n_notes + 1):
            c = cost[wi][ni]
            if c == inf:
                continue
            if wi < n_words and ni < n_notes:
                d = abs(words[wi][0] - float(note_onsets[ni]))
                if d <= _DP_MAX_MATCH_DIST:
                    mc = c + d + (0.0 if note_presence[ni] else _DP_REST_PROMOTE_PENALTY)
                    if mc < cost[wi + 1][ni + 1]:
                        cost[wi + 1][ni + 1] = mc
                        back[wi + 1][ni + 1] = ("m", wi, ni)
            if wi < n_words:
                sc = c + _DP_SKIP_WORD_COST
                if sc < cost[wi + 1][ni]:
                    cost[wi + 1][ni] = sc
                    back[wi + 1][ni] = ("sw", wi, ni)
            if ni < n_notes:
                if c < cost[wi][ni + 1]:
                    cost[wi][ni + 1] = c
                    back[wi][ni + 1] = ("sn", wi, ni)

    matched: dict = {}  # word_idx -> note_idx
    skipped_words = []
    wi, ni = n_words, n_notes
    while (wi, ni) != (0, 0):
        move, pwi, pni = back[wi][ni]
        if move == "m":
            matched[wi - 1] = ni - 1
        elif move == "sw":
            skipped_words.append(wi - 1)
        wi, ni = pwi, pni

    n_promoted = sum(1 for w_idx, n_idx in matched.items() if not note_presence[n_idx])
    n_carried = len(skipped_words)

    for w_idx, n_idx in matched.items():
        assigned[n_idx] = words[w_idx][1]

    if skipped_words:
        # carry skipped characters: forward onto the next matched note's
        # lyric, else backward onto the last matched note — never dropped
        matched_word_idxs = sorted(matched)
        carry_prefix: dict = {}
        carry_suffix: dict = {}
        for sw in sorted(skipped_words):
            nxt = next((m for m in matched_word_idxs if m > sw), None)
            if nxt is not None:
                carry_prefix[nxt] = carry_prefix.get(nxt, "") + words[sw][1]
            elif matched_word_idxs:
                prv = matched_word_idxs[-1]
                carry_suffix[prv] = carry_suffix.get(prv, "") + words[sw][1]
            else:
                # degenerate: nothing matched at all — park the character on
                # the nearest note rather than losing it
                onset, char = words[sw]
                near = int(np.argmin(np.abs(note_onsets - onset)))
                assigned[near] = char if assigned[near] in ("", "-") else assigned[near] + char
        for w_idx, prefix in carry_prefix.items():
            assigned[matched[w_idx]] = prefix + assigned[matched[w_idx]]
        for w_idx, suffix in carry_suffix.items():
            assigned[matched[w_idx]] = assigned[matched[w_idx]] + suffix

    return assigned, n_promoted, n_carried


def extract_pitches_and_align_torch(
    chunks,
    sr,
    pred_dict,
    chars_dict,
    game_model,
    device,
    ts,
    seg_threshold,
    seg_radius,
    est_threshold,
    batch_size=4,
    debug_mode=False,
    cancel_checker=None,
    language=None,
):
    """
    Extract pitches using the GAME ONNX runtime and align them to lyrics.
    """
    del device, debug_mode
    print("[Hybrid Pipeline] Extracting pitches with GAME ONNX...")

    all_notes = []
    batch_infos = []
    processed_chunk_indices = set()

    for chunk_idx, chunk in enumerate(chunks):
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        stem = f"chunk_{chunk_idx}"
        if stem not in pred_dict:
            print(f"[Warning] {stem}: missing HFA prediction; skipping lyric-aligned GAME for this chunk.")
            continue

        _, _, result_word = pred_dict[stem]
        if not result_word:
            print(f"[Warning] {stem}: empty HFA word result; skipping lyric-aligned GAME for this chunk.")
            continue

        word_durs, word_vuvs, lyrics = extract_vowel_boundaries(
            result_word,
            chars_dict.get(stem, []),
            language=language,
        )
        if not word_durs:
            print(f"[Warning] {stem}: no usable word durations; skipping lyric-aligned GAME for this chunk.")
            continue

        batch_infos.append(
            {
                "chunk_idx": chunk_idx,
                "waveform": chunk["waveform"],
                "waveform_duration": len(chunk["waveform"]) / sr,
                "word_durs": word_durs,
                "known_durations": np.asarray(word_durs, dtype=np.float32),
                "offset": chunk["offset"],
                "word_vuvs": word_vuvs,
                "lyrics": lyrics,
            }
        )

    for i in range(0, len(batch_infos), batch_size):
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        batch = batch_infos[i : i + batch_size]

        waveforms_np = [info["waveform"] for info in batch]
        waveform_durations_np = [info["waveform_duration"] for info in batch]
        known_durations_np = [info["known_durations"] for info in batch]

        try:
            batch_results = _run_game_inference_batch(
                game_model=game_model,
                waveforms_np=waveforms_np,
                waveform_durations_np=waveform_durations_np,
                known_durations_np=known_durations_np,
                seg_threshold=seg_threshold,
                seg_radius=seg_radius,
                est_threshold=est_threshold,
                ts=ts,
                language=language,
            )
        except Exception:
            print("Error during GAME ONNX inference batch:")
            traceback.print_exc()
            raise

        for result, info in zip(batch_results, batch):
            durations, presence, scores = result

            note_dur = durations[durations > 0].tolist()
            valid_presence = presence[durations > 0]
            valid_scores = scores[durations > 0]
            if not note_dur:
                print(f"[Warning] GAME returned no note durations for chunk at {info['offset']:.2f}s; skipping.")
                continue

            note_seq = [
                librosa.midi_to_note(float(m), unicode=False, cents=True) if v else "rest"
                for m, v in zip(valid_scores, valid_presence)
            ]

            # GAME's audio-driven note timeline is the rhythm ground truth:
            # the (repaired) HFA word grid can still skew against it by
            # hundreds of ms, and cropping notes to the grid is what produced
            # the crushed/slurred melody. Pass GAME notes through untouched
            # and hang characters on them with a monotonic DP alignment.
            valid_scores_l = [float(s) for s in valid_scores]
            note_lyrics, n_promoted, n_carried = _assign_lyrics_dp(
                info["word_durs"],
                info["word_vuvs"],
                info["lyrics"],
                note_dur,
                [bool(v) for v in valid_presence],
            )
            if n_promoted or n_carried:
                print(
                    f"[Lyric DP] chunk@{info['offset']:.2f}s: promoted {n_promoted} "
                    f"rest->voiced, carried {n_carried} unmatched word(s)"
                )

            current_onset = info["offset"]
            for n_idx, (n_seq, n_dur) in enumerate(zip(note_seq, note_dur)):
                lyric_to_assign = note_lyrics[n_idx]
                if n_seq == "rest" and lyric_to_assign in ("", "-"):
                    current_onset += n_dur
                    continue

                if n_seq == "rest":
                    # word matched a rest note GAME judged unvoiced, but it
                    # carries a character -> promote to voiced with GAME's
                    # pitch estimate for it
                    pitch = valid_scores_l[n_idx]
                else:
                    pitch = librosa.note_to_midi(n_seq, round_midi=False)

                is_contiguous = len(all_notes) > 0 and abs(all_notes[-1].offset - current_onset) < 0.01
                can_merge = is_contiguous and abs(all_notes[-1].pitch - pitch) < 0.1 and lyric_to_assign in ("", "-")

                if can_merge:
                    all_notes[-1].offset += n_dur
                else:
                    all_notes.append(
                        NoteInfo(
                            onset=current_onset,
                            offset=current_onset + n_dur,
                            pitch=pitch,
                            lyric=lyric_to_assign if lyric_to_assign else "-",
                        )
                    )

                current_onset += n_dur
            processed_chunk_indices.add(info["chunk_idx"])

    return all_notes, processed_chunk_indices


def extract_pitches_only_torch(
    chunks,
    sr,
    game_model,
    device,
    ts,
    seg_threshold,
    seg_radius,
    est_threshold,
    batch_size=4,
    debug_mode=False,
    cancel_checker=None,
    language=None,
):
    """
    Extract pitches using the GAME ONNX runtime without lyric alignment.
    """
    del device, debug_mode
    print("[Hybrid Pipeline] Extracting pitches with GAME ONNX (no-lyrics mode)...")

    all_notes = []
    batch_infos = []
    for chunk in chunks:
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        batch_infos.append(
            {
                "waveform": chunk["waveform"],
                "waveform_duration": len(chunk["waveform"]) / sr,
                "offset": chunk["offset"],
            }
        )

    for i in range(0, len(batch_infos), batch_size):
        if cancel_checker and cancel_checker():
            raise InterruptedError("GAME task cancelled")
        batch = batch_infos[i : i + batch_size]

        waveforms_np = [info["waveform"] for info in batch]
        waveform_durations_np = [info["waveform_duration"] for info in batch]
        known_durations_np = [np.zeros(1, dtype=np.float32) for _ in batch]

        try:
            batch_results = _run_game_inference_batch(
                game_model=game_model,
                waveforms_np=waveforms_np,
                waveform_durations_np=waveform_durations_np,
                known_durations_np=known_durations_np,
                seg_threshold=seg_threshold,
                seg_radius=seg_radius,
                est_threshold=est_threshold,
                ts=ts,
                language=language,
            )
        except Exception:
            print("Error during GAME ONNX inference batch (no-lyrics):")
            traceback.print_exc()
            raise

        for result, info in zip(batch_results, batch):
            durations, presence, scores = result
            valid = durations > 0
            note_dur = durations[valid].tolist()
            note_presence = presence[valid].tolist()
            note_scores = scores[valid].tolist()
            if not note_dur:
                print(f"[Warning] GAME returned no note durations for chunk at {info['offset']:.2f}s; skipping.")
                continue

            current_onset = info["offset"]
            for n_dur, n_presence, n_score in zip(note_dur, note_presence, note_scores):
                if n_presence:
                    all_notes.append(
                        NoteInfo(
                            onset=current_onset,
                            offset=current_onset + n_dur,
                            pitch=float(n_score),
                            lyric="",
                        )
                    )
                current_onset += n_dur

    return all_notes


def extract_pitches_and_align_onnx(*args, **kwargs):
    return extract_pitches_and_align_torch(*args, **kwargs)


def extract_pitches_only_onnx(*args, **kwargs):
    return extract_pitches_only_torch(*args, **kwargs)
