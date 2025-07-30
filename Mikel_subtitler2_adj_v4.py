import os
import torch
import whisperx
import spacy
import re

LOCUCIONES_CONJUNTIVAS = {
    ("a", "medida", "que"),
    ("a", "pesar", "de"),
    ("al", "igual", "que"),
    ("así", "como"),
    ("aunque",),
    ("con", "el", "fin", "de"),
    ("con", "el", "objeto", "de"),
    ("con", "tal", "que"),
    ("con", "todo"),
    ("como", "si"),
    ("como", "que"),
    ("de", "ahí", "que"),
    ("de", "modo", "que"),
    ("de", "forma", "que"),
    ("de", "manera", "que"),
    ("en", "cambio"),
    ("en", "caso", "de", "que"),
    ("en", "cuanto"),
    ("en", "la", "medida", "en", "que"),
    ("en", "vez", "de"),
    ("mientras", "que"),
    ("no", "obstante"),
    ("para", "que"),
    ("por", "consiguiente"),
    ("por", "eso"),
    ("por", "lo", "tanto"),
    ("por", "más", "que"),
    ("puesto", "que"),
    ("salvo", "que"),
    ("siempre", "que"),
    ("sin", "embargo"),
    ("tan", "pronto", "como"),
    ("toda", "vez", "que"),
    ("ya", "que"),
}


class Token:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text

    def __str__(self):
        return self.text


class Subtitle:
    def __init__(self):
        self.tokens = []
        self.segments = []  # segmentos originales de Whisper
        self.nlp = spacy.load("es_core_news_lg")

    def add_token(self, token):
        self.tokens.append(token)

    def set_segments(self, segments):
        self.segments = []

        for seg in segments:
            text = seg["text"].strip()
            splits = re.split(r"(?<=[\.\?])\s+", text)
            for s in splits:
                s = s.strip()
                if s:  # descarta fragmentos vacíos
                    self.segments.append({"text": s, "start": seg["start"]})

                # # Dividir por punto si hay varios en una misma oración
                # sub_sentences = [s.strip() for s in sentence.split(".") if s.strip()]
                # if sub_sentences:
                #     for s in sub_sentences:
                #         if not s.endswith(".") and not s.endswith("?"):
                #             s += "."
                #         self.segments.append({"text": s})
                # else:
                #     self.segments.append({"text":sentence})

            # Si spaCy no detecta nada (extremo raro), añadir el texto completo
            if not self.segments:
                self.segments.append({"text": text, "start": seg["start"]})

    def export_vtt(self, output_file):
        blocks = self._build_blocks_from_segments()
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for block in blocks:
                f.write(
                    f"{self._format_time(block['start'])} --> {self._format_time(block['end'])}\n"
                )
                f.write(block["text"].strip() + "\n\n")

    def _build_blocks_from_segments(self):
        blocks = []

        for segment in self.segments:
            remaining_tokens = [t for t in self.tokens if t.start >= segment["start"]]
            segment_tokens = match_tokens_to_text(remaining_tokens, segment["text"])
            if not segment_tokens:
                print(segment["end"])
                print(
                    "No se ha conseguido juntar el segmento al token", segment["text"]
                )
                continue

            i = 0
            while i < len(segment_tokens):
                block_start = segment_tokens[i].start
                current_tokens = []
                best_lines = None
                best_score = -20
                best_end_index = None

                for j in range(i, len(segment_tokens)):
                    segment_text = "".join(
                        t.text
                        for t in segment_tokens[
                            max(0, i - 5) : min(len(segment_tokens), j + 5)
                        ]
                    )
                    current_tokens.append(segment_tokens[j])
                    current_text = "".join(t.text for t in current_tokens)

                    # Hasta que no llegemos a los 37 caracteres no nos hace falta hacer split_by_syntagmas
                    if len(current_text) <= 37:
                        best_lines = [current_text, " "]
                        best_score = -15
                        best_end_index = j
                        continue

                    score, lines = self._split_by_syntagmas(current_text, segment_text)

                    if eval_une_4_3("\n".join(lines)) and all(
                        eval_une_4_6(line) for line in lines
                    ):
                        if score > best_score:
                            best_score = score
                            best_lines = lines
                            best_end_index = j
                        continue
                    if len(current_text) >= 75:
                        break

                if best_lines is not None:

                    end_token = segment_tokens[best_end_index]
                    duration = end_token.end - block_start
                    comply, min_dur = eval_une_5_1(" ".join(best_lines), duration)
                    end_time = end_token.end if comply else block_start + min_dur
                    text = "\n".join(best_lines)
                    if len(text) <= 37:
                        text = " ".join(best_lines)
                    if blocks:
                        last_block = blocks[-1]
                        if block_start < last_block["end"]:
                            if block_start > last_block["start"]:
                                last_block["end"] = block_start
                            elif end_time > last_block["end"]:
                                block_start = last_block["end"]
                            else:
                                print(
                                    "no se ha podido poner subtitulado sin solapamiento en el bloque: ",
                                    block_start,
                                    "-",
                                    end_time,
                                )

                    blocks.append(
                        {
                            "text": text,
                            "start": block_start,
                            "end": end_time,
                        }
                    )
                    i = best_end_index + 1
                else:
                    valid_tokens = (
                        current_tokens[:-1]
                        if len(current_tokens) > 1
                        else current_tokens
                    )
                    valid_text = "".join(t.text for t in valid_tokens)
                    score, lines = self._split_by_syntagmas(valid_text, segment_text)
                    duration = valid_tokens[-1].end - block_start
                    comply, min_dur = eval_une_5_1(valid_text, duration)
                    end_time = valid_tokens[-1].end if comply else block_start + min_dur
                    text = "\n".join(lines)
                    if len(text) <= 37:
                        text = " ".join(lines)
                    if blocks:
                        last_block = blocks[-1]
                        if block_start < last_block["end"]:
                            if block_start > last_block["start"]:
                                last_block["end"] = block_start
                            elif end_time > last_block["end"]:
                                block_start = last_block["end"]
                        else:
                            print(
                                "no se ha podido poner subtitulado sin solapamiento en el bloque: ",
                                block_start,
                                "-",
                                end_time,
                            )
                    blocks.append(
                        {
                            "text": text,
                            "start": block_start,
                            "end": end_time,
                        }
                    )
                    i += len(valid_tokens)

        return blocks

    def _split_by_syntagmas(self, text, segment_text):

        max_chars = 37
        words = text.strip().split()
        num_words = len(words)

        doc = self.nlp(segment_text.strip())
        doc = [tok for tok in doc if not tok.is_punct]

        cut_words = {
            "con",
            "y",
            "o",
            "pero",
            "aunque",
            "ni",
            "sino",
            "de",
            "del",
            "la",
            "el",
            "al",
            "un",
            "lo",
            "los",
            "las",
            "les",
            "le",
            "la",
            "para",
            "por",
            "que",
            "a",
            "se",
            "en",
        }
        if words[-1] in cut_words:
            score = -2
        else:
            score = 0

        if len(text) <= max_chars:
            right_last_token = None
            right = words
            # Buscar los tokens que coincidan con la última palabra de left y right

            right_target = clean_word(right[-1])  # primera palabra de right
            right_needed = [clean_word(w) for w in segment_text.split()].count(
                right_target
            )
            j, right_last_token = find_nth_token(doc, right_target, right_needed)

            if right_last_token is not None:

                left_tokens = doc[: j + 1]
                right_tokens = doc[j + 1 :]

                if right_tokens:
                    score += negative_scores(left_tokens, right_tokens)
                    score += corta_locucion_tokens(left_tokens, right_tokens)

            return (score + 0.4 * num_words, [text, " "])

        candidates = []

        for i in range(1, num_words - 1):
            left = words[:i]
            right = words[i:]
            left_words = " ".join(left)
            right_words = " ".join(right)
            if len(left_words) <= max_chars and len(right_words) <= max_chars:
                score_i = score + 0.4 * len(left) + 0.4 * len(right)
                if left[-1][-1] == "," or right[-1][-1] == ",":
                    score_i += 0.5
                minus_difference = 0.1 * abs(len(left) - len(right))
                candidates.append((score_i - minus_difference, [left, right]))

        if not candidates:
            return -20, ["", ""]
        candidates_division_sintagmas = []
        for score, [left, right] in candidates:
            if not left or not right:
                continue
            left_last_token = None
            right_last_token = None

            # Buscar los tokens que coincidan con la última palabra de left y right

            left_target = clean_word(left[-1])  # última palabra de left
            right_target = clean_word(right[-1])  # primera palabra de right

            left_needed = [clean_word(w) for w in segment_text.split()].count(
                left_target
            )
            right_needed = [clean_word(w) for w in segment_text.split()].count(
                right_target
            )

            i, left_last_token = find_nth_token(doc, left_target, left_needed)
            j, right_last_token = find_nth_token(doc, right_target, right_needed)

            if i is not None:

                if left_last_token.text in cut_words:
                    score -= 2

                # Evaluar si corta locución en tokens
                left_tokens = doc[: i + 1]
                right_tokens = doc[i + 1 :]

                if right_tokens:
                    score += negative_scores(left_tokens, right_tokens)
                    score += corta_locucion_tokens(left_tokens, right_tokens)

            if j is not None:

                left_tokens = doc[max(0, j - 4) : j + 1]
                right_tokens = doc[j + 1 :]
                if right_tokens:
                    score += negative_scores(left_tokens, right_tokens)
                    score += corta_locucion_tokens(left_tokens, right_tokens)

            candidates_division_sintagmas.append(
                (score, [" ".join(left), " ".join(right)])
            )

        if candidates_division_sintagmas:
            candidates_division_sintagmas.sort(reverse=True, key=lambda x: x[0])
            return candidates_division_sintagmas[0]
        else:
            print("ALGO NO HA COGIDO")
        return -20, ["", ""]

    def _format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        # s=max(0,s-0.3)
        return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


def match_tokens_to_text(tokens, target_text):
    """
    Busca una secuencia de tokens que reconstruya target_text, comenzando
    desde cada coincidencia de la primera palabra.
    """
    target_words = target_text.strip().split()
    n = len(target_words)
    target_joined = "".join(target_words).lower()

    for i, token in enumerate(tokens):
        if token.text.strip().lower() == target_words[0].lower():
            candidate_tokens = tokens[i : i + n]
            candidate_joined = "".join(t.text.strip() for t in candidate_tokens).lower()
            if candidate_joined == target_joined:
                return candidate_tokens

    return []


# Reglas UNE


def eval_une_4_3(sentence):
    return sentence.strip().count("\n") < 2


def eval_une_4_6(line):
    return len(line.strip()) <= 37


def eval_une_5_1(sentence, duration):
    char_count = len(sentence.replace("\n", ""))
    cps = char_count / duration if duration > 0 else 0
    min_duration = char_count / 15.0
    return cps <= 15.0, min_duration


def prep_negative_score(token):
    if token.dep_ == "case" or token.dep_ == "det":
        return -3
    else:
        return 0


def conj_negative_score(token):
    if token.dep_ == "cc" or token.dep_ == "mark" or token.dep_ == "advmod":
        return -3
    else:
        return 0


def verb_negative_score(token):
    if token.pos_ == "AUX" or token.dep_ == "cop":
        return -5
    else:
        return 0


def corta_locucion_tokens(left_tokens, right_tokens, locuciones=LOCUCIONES_CONJUNTIVAS):
    """
    Penaliza si el corte divide una locución conjuntiva conocida (parte en left, parte en right),
    pero no penaliza si la locución aparece completamente en right_tokens.
    """
    if not left_tokens or not right_tokens:
        return 0

    # Obtener textos en minúsculas
    left_texts = [t.text.lower() for t in left_tokens[-3:]]
    right_texts = [t.text.lower() for t in right_tokens[:3]]

    max_len = max(len(loc) for loc in locuciones)

    # Solo penalizar si la locución está fragmentada entre left y right
    for n in range(2, max_len + 1):
        for i in range(1, n):  # posición del corte dentro de la locución
            left_part = tuple(left_texts[-i:]) if i <= len(left_texts) else ()
            right_part = (
                tuple(right_texts[: n - i]) if (n - i) <= len(right_texts) else ()
            )

            combined = left_part + right_part

            if combined in locuciones:
                return -5  # penalización por cortar una locución

    return 0


def penaliza_dos_verbos(token, token_2):
    """
    Devuelve -5 si ambos tokens son verbos (VERB o AUX), 0 en caso contrario.
    """
    if token.pos_ in {"VERB", "AUX"} and token_2.pos_ in {"VERB", "AUX"}:
        return -5
    return 0


def penaliza_flat(token1, token2):
    if token1.dep_ == "flat" and token2.head == token1.head:
        return -4
    elif token2.dep_ == "flat" and token1.head == token2.head:
        return -4
    elif token1.head == token2 or token2.head == token1:
        if token1.dep_ == "flat" or token2.dep_ == "flat":
            return -4
    return 0


def amod_negative_score(left_tokens, right_tokens):
    """
    Penaliza si el corte separa un adjetivo (amod) de su sustantivo (head):
      1) Último token de la izquierda es 'amod' y su head está en la derecha.
      2) Primer token de la derecha es 'amod' y su head está en la izquierda.
    """
    if not left_tokens or not right_tokens:
        return 0

    last_left = left_tokens[-1]
    first_right = right_tokens[0]

    # 1️⃣ amod en la izquierda, head en la derecha
    if last_left.dep_ == "amod" and last_left.head in right_tokens:
        return -3

    # 2️⃣ amod en la derecha, head en la izquierda
    if first_right.dep_ == "amod" and first_right.head in left_tokens:
        return -3

    return 0


def head_negative_score(left_tokens, right_tokens):
    """
    Penaliza si el corte separa un adjetivo (amod) de su sustantivo (head):
      1) Último token de la izquierda es 'amod' y su head está en la derecha.
      2) Primer token de la derecha es 'amod' y su head está en la izquierda.
    """
    if not left_tokens or not right_tokens:
        return 0

    last_left = left_tokens[-1]
    first_right = right_tokens[0]

    if last_left.head == first_right:
        return -3

    if first_right.head == last_left:
        return -3

    if last_left.head in right_tokens:
        return -2

    if first_right.head in left_tokens:
        return -2

    return 0


def negative_scores(left_tokens, right_tokens):
    token = left_tokens[-1]
    token_2 = right_tokens[0]
    verbos = verb_negative_score(token)
    negativos = prep_negative_score(token) + verbos + conj_negative_score(token)
    # si la primera condición de que el primero sea verbo auxiliar no da resultado, se comprueba que no sean dos verbos
    if verbos == 0 and token_2 is not None:
        negativos += penaliza_dos_verbos(token, token_2)

    if token_2 is not None:
        negativos += penaliza_flat(token, token_2)
        negativos += amod_negative_score(left_tokens, right_tokens)
        negativos += head_negative_score(left_tokens, right_tokens)
    return negativos


def interpolate_missing_timestamps(words):
    new_words = []
    last_time = None

    for i, word_data in enumerate(words):
        if "start" in word_data and "end" in word_data:
            new_words.append(word_data)
            last_time = (word_data["start"], word_data["end"])
        else:
            j = i + 1
            while j < len(words) and ("start" not in words[j] or "end" not in words[j]):
                j += 1
            next_time = (
                (words[j]["start"], words[j]["end"]) if j < len(words) else last_time
            )

            if last_time and next_time:
                start = max(last_time[1], (last_time[1] + next_time[0]) / 2 - 0.01)
                end = (last_time[1] + next_time[0]) / 2 + 0.01
                word_data["start"] = max(0, start)
                word_data["end"] = end
            else:
                word_data["start"] = 0
                word_data["end"] = 0.1
            new_words.append(word_data)

    return new_words


def find_nth_token(doc, target_text, n):
    """
    Devuelve el token que corresponde a la n-ésima aparición de `target_text`
    (conteo desde el principio del Doc, n ≥ 1).  Si no se encuentra, None.
    """
    target_text = target_text.lower()
    count = 0
    for i, tok in enumerate(doc):
        if clean_word(tok.text) == target_text:
            count += 1
            if count == n:
                return i, tok
    return None, None


def clean_word(w: str) -> str:
    """
    Devuelve la palabra en minúsculas sin puntuación inicial/final.
    Ej.: "náuseas," -> "náuseas"
    """
    return re.sub(r"^[\W_]+|[\W_]+$", "", w.lower())


def generate_vtt_from_audio(
    audio_path,
    output_vtt_path,
    model_name="large-v2",
    language="es",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("→ Cargando WhisperX...")
    model = whisperx.load_model(
        model_name, device, language=language, compute_type="int8_float16"
    )
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=8)

    print("→ Alineando palabras...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
    aligned["word_segments"] = interpolate_missing_timestamps(aligned["word_segments"])

    print("→ Construyendo subtítulos...")
    sub = Subtitle()
    sub.set_segments(result["segments"])

    for word in aligned["word_segments"]:
        if all(k in word for k in ("start", "end", "word")):
            sub.add_token(Token(word["start"], word["end"], word["word"] + " "))

    sub.export_vtt(output_vtt_path)
    print(f"✔ Subtítulo exportado: {output_vtt_path}")
