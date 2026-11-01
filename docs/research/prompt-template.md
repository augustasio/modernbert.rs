# ModernBERT-Large-Instruct: exact MLM-head zero-shot classification prompt format

Research date: 2026-11-01. No code changes made.

## Sources read

| # | Source | URL |
|---|--------|-----|
| S1 | HF model card (raw `README.md`, not the rendered page) | https://huggingface.co/answerdotai/ModernBERT-Large-Instruct/raw/main/README.md |
| S2 | Paper (arXiv HTML v2), §2.5.1 Training / §2.5.2 Inference | https://arxiv.org/html/2502.03793v2 |
| S3 | Eval harness `eval.py` (MMLU) | https://raw.githubusercontent.com/AnswerDotAI/ModernBERT-Instruct-mini-cookbook/main/eval.py |
| S4 | `train.py` (AG News fine-tune template) | https://raw.githubusercontent.com/AnswerDotAI/ModernBERT-Instruct-mini-cookbook/main/train.py |
| S5 | `sweep_20newsgroups.py` (letter-verbalizer classification template) | https://raw.githubusercontent.com/AnswerDotAI/ModernBERT-Instruct-mini-cookbook/main/sweep_20newsgroups.py |
| S6 | Answer.AI blog post | https://answer.ai/posts/2025-02-10-modernbert-instruct.html |
| S7 | Tokenizer files, inspected locally with `tokenizers` | `.../raw/main/tokenizer.json`, `.../raw/main/tokenizer_config.json` |

The repo is `AnswerDotAI/ModernBERT-Instruct-mini-cookbook` (linked from the model card). It contains
`eval.py`, `train.py`, `sweep_20newsgroups.py`, `agnews_examples/agnews_label_map.json`.

**Note on method:** `WebFetch` refused to reproduce the model card and paper verbatim (it has a
125-character quote cap). All literal strings below come from `curl` of the raw `README.md` /
raw GitHub files, with byte-exactness checked via `cat -vet` and via Python `ast` extraction of the
string literals. The tokenizer claims are verified by running the real tokenizer, not inferred.

---

## 1. The literal template strings

There is **no single canonical template**. The sources carry three different ones, and the harness
template differs from the model card's. Reported separately, as requested.

### S3 — `eval.py`, the MMLU eval harness (most reliable; this is the literal string the reported MMLU number came from)

Constructed by `format_mmlu_example`:

```python
mc_prefix = (
    "You will be given a question as well as a list of options. "
    "Read the question carefully and select the right answer from the list.\n"
    "QUESTION:\n"
)
letters = ['A', 'B', 'C', 'D']
...
text = mc_prefix + question + "\nCHOICES:\n"
for i, choice in enumerate(choices):
    text += f"- {letters[i]}: {choice}\n"
text += "\nANSWER:\nAnswer: [unused0] [MASK]"
```

Rendered (verified by running the code; blank line before `ANSWER:` is real, from the trailing
`\n` of the last choice plus the leading `\n` of `"\nANSWER:"`):

```
You will be given a question as well as a list of options. Read the question carefully and select the right answer from the list.
QUESTION:
An expected side effect of creatine supplementation is:
CHOICES:
- A: muscle weakness.
- B: gain in body mass.
- C: muscle cramps.
- D: loss of electrolytes.

ANSWER:
Answer: [unused0] [MASK]
```

As a Python repr, so no whitespace is ambiguous:

```
'You will be given a question as well as a list of options. Read the question carefully and select the right answer from the list.\nQUESTION:\nAn expected side effect of creatine supplementation is:\nCHOICES:\n- A: muscle weakness.\n- B: gain in body mass.\n- C: muscle cramps.\n- D: loss of electrolytes.\n\nANSWER:\nAnswer: [unused0] [MASK]'
```

Points to note: the instruction is **one line** (the two Python string fragments concatenate with a
single space, no newline). `QUESTION:` and `CHOICES:` and `ANSWER:` each sit on their own line with
the content on the *following* line. Each choice line ends with `\n`, so there is a **blank line**
between the last choice and `ANSWER:`. The word `Answer:` appears a second time on the line after
`ANSWER:`.

### S1/S6 — HF model card and blog post (differs from the harness)

Verbatim from raw `README.md` (identical string in the blog post and in the `widget:` YAML, modulo
the widget's unquoted newline):

```
You will be given a question and options. Select the right answer.
QUESTION: If (G, .) is a group such that (ab)^-1 = a^-1b^-1, for all a, b in G, then G is a/an
CHOICES:
- A: commutative semi group
- B: abelian group
- C: non-abelian group
- D: None of these
ANSWER: [unused0] [MASK]
```

Differences from S3: shorter instruction (`"...a question and options. Select the right answer."`);
question on the **same** line as `QUESTION:`; **no blank line** before `ANSWER:`; and crucially
**no second `Answer:` prefix** — it is `ANSWER: [unused0] [MASK]` on one line.

### S2 — paper §2.5.2 Inference

The paper's rendered listing matches **S3**, not the model card: it uses the long instruction
(`"You will be given a question as well as a list of options. Read the question carefully and
select the right answer from the list."`), `CHOICES:` with `- A:` bullets, and ends:

```
ANSWER:

Answer: [unused0] [MASK]
```

i.e. it has the doubled `ANSWER:` / `Answer:` of the harness. (The HTML-to-text extraction inserts
blank lines between every block, so the paper cannot settle exact newline counts on its own; S3 can,
and they agree on content.)

The paper also describes the structure in prose:

> "(1) basic instructions are given to the model, followed by (2) the text to be labeled or
> responded to, (3) the list of potential labels, and finally, (4) an answer section that ends in a
> [MASK] token"

Training (§2.5.1) adds no instruction line: FLAN's own templates are reused, and the only
templating step is prepending `[unused0]` to the `[MASK]`. The paper's training example ends:

```
Which language is this?
[unused0] [MASK]
```

### S4 — `train.py`, the AG News template (verbose labels, not letters)

This is the one actually used for the AG News number. Extracted via `ast` from the source, so the
**4 trailing spaces after `** Instructions **` are real** (confirmed independently with `cat -vet`):

```
** Instructions **    
Given a short news report, determine which news category it belongs to.
** Content **
What is the category of the following news report?
News Report: {txt}
** Options **
- Business: It is about business or financial news.
- World: It's about world news.
- Sports: It's about sports.
- Tech: It's about Science or Technology, or related companies.
** Answer **
Answer: [unused0] {label}
```

repr (note `**    \n`):

```
"** Instructions **    \nGiven a short news report, determine which news category it belongs to.\n** Content **\nWhat is the category of the following news report?\nNews Report: {txt}\n** Options **\n- Business: It is about business or financial news.\n- World: It's about world news.\n- Sports: It's about sports.\n- Tech: It's about Science or Technology, or related companies.\n** Answer **\nAnswer: [unused0] {label}"
```

`{label}` is filled from `agnews_examples/agnews_label_map.json`:

```json
{
    "Business": "Business",
    "World": "World",
    "Sports": "Sports",
    "Sci/Tech": "Tech"
}
```

So AG News, in the authors' own code, uses **semantic single-token verbalizers**
(`Business`/`World`/`Sports`/`Tech`), **not** `A`/`B`/`C`/`D`. The `Sci/Tech` -> `Tech` remap exists
because `Sci/Tech` is 3 tokens. Note the section headers are `** Options **` / `** Answer **`, and
option lines are `- Label: description.` — the label itself is the verbalizer, the description is
gloss text. At inference the `{label}` slot becomes `[MASK]`.

The repo README confirms letters are the general-purpose fallback:

> "In AGNews, we use verbose labels, but it works really well by using letter prefixes for each
> answers! In fact, this is how we perform most tasks."

### S5 — `sweep_20newsgroups.py`, the letter-verbalizer classification template

The closest thing in the repo to a letter-based *classification* (not multiple-choice-QA) template.
Same `** ... **` skeleton as S4:

```
** Instructions **    
Given a short post from an online forum, determine which newsgroup category it belongs to.
** Content **
{text_content}
What's the category of the post above?
** Options **
{options}
** Answer **
Answer: [unused0] {label_letter}
```

with

```python
options = "\n".join([f"{letter}: {label}, {desc}" for label, (letter, desc) in label_descriptions.items()])
```

So here the option lines are `A: alt.atheism, Discussion of atheism and criticism of religion` —
**no leading `- `**, and the format is `LETTER: raw_label, description`, newline-separated. This
disagrees with the `- A: text` form in S1/S2/S3.

---

## 2. Where `[MASK]` sits, and how many

**Exactly one `[MASK]`**, always the final token of the prompt text, always immediately preceded by
`[unused0]` and a single space: the literal tail is `[unused0] [MASK]`.

Verified by running the tokenizer on the S3 and S4 prompts: `ids.count(50284) == 1` for both.

Relevant ids (from `tokenizer.json`): `[MASK]` = **50284**, `[unused0]` = **50285**,
`[CLS]` = 50281, `[SEP]` = 50282.

`[unused0]`'s role, from S2 §2.5.1:

> "the [unused0] token present in the ModernBERT vocabulary and left purposefully untrained"

> "indicating that the following token is the model's final prediction"

It is a *non-special*, `normalized: true` added token (id 50285), so it survives as one token in
ordinary text. Tokenized tail of the S3 prompt:

```
['SW', 'ER', ':', 'Ċ', 'Answer', ':', 'Ġ', '[unused0]', ' [MASK]', '[SEP]']
```

Worth knowing: `[MASK]` has `lstrip: true`, so it swallows the space before it. The space that would
precede the answer letter is therefore *not* a separate token in the masked prompt, and the `[MASK]`
sits at exactly the position the answer token occupied during training. The `'Ġ'` (id 209) before
`[unused0]` is a standalone space token, because `[unused0]` has `lstrip: false`.

The harness locates the mask by id and takes the **last** occurrence:

```python
mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
mask_logits = outputs.logits[0, mask_indices[-1, 1]]
```

(The model card instead takes `[0, 1]`, the first occurrence. Irrelevant with one mask.)

---

## 3. Option formatting

- S1 (model card), S2 (paper), S3 (`eval.py`): **`- A: text`** — a hyphen, a space, the capital
  letter, a colon, a space, the option text. One option per line, **newline-separated**, each line
  terminated by `\n`. Preceded by a line reading exactly `CHOICES:`.
- S4 (`train.py`, AG News): **`- Business: description.`** — same `- X: ` shape but `X` is the
  semantic label word, under a `** Options **` header.
- S5 (`sweep_20newsgroups.py`): **`A: label, description`** — **no leading `- `**, under
  `** Options **`.

None of the sources use `A)` or `(A)`. Nothing uses comma- or space-separated options; it is always
one option per line.

---

## 4. Which tokens the logits are compared at — the byte-level BPE question

**Answer: the space-prefixed variants (`ĠA`), i.e. ids 329 / 378 / 330 / 399 — not the bare
`A` = 34.** And this happens *automatically*, because the tokenizer sets `add_prefix_space: true`.
This is the one place where a naive reimplementation silently breaks, so I verified it by running
the real tokenizer rather than reasoning about it.

Verbatim, the full option-resolution code from S3 `eval.py` (`evaluate_mmlu`, lines 111-119, and
`predict_answer_mlm`, lines 84-91):

```python
    # Deprecated but kept for legacy reasons pre-tokenizer fix.
    option_mapping = {
        "A": "A", " A": "A",
        "B": "B", " B": "B",
        "C": "C", " C": "C",
        "D": "D", " D": "D",
    }
    option_tokens = ["A", " A", "B", " B",
                     "C", " C", "D", " D",]
```

```python
    # Compute softmax probabilities over the vocabulary.
    probabilities = torch.nn.functional.softmax(mask_logits, dim=-1)
    
    # For each candidate option, get the probability of its token.
    option_ids = [tokenizer.encode(t)[1] for t in option_tokens]
    option_probs = {}
    for token, token_id in zip(option_tokens, option_ids):
        canonical = option_mapping[token]
        option_probs[canonical] = option_probs.get(canonical, 0) + probabilities[token_id].item()
    
    # Select the option (letter) with the highest probability.
    predicted_token = max(option_probs.items(), key=lambda x: x[1])[0]
```

`tokenizer.encode(t)[1]` — index `[1]` skips `[CLS]`.

Measured behaviour of the real tokenizer (`pre_tokenizer: ByteLevel, add_prefix_space: true`):

```
'A'   ids=[50281, 329, 50282]  tokens=['[CLS]', 'ĠA', '[SEP]']   -> [1] = 329
' A'  ids=[50281, 329, 50282]  tokens=['[CLS]', 'ĠA', '[SEP]']   -> [1] = 329
'B'   -> 378 ('ĠB');  ' B' -> 378
'C'   -> 330 ('ĠC');  ' C' -> 330
'D'   -> 399 ('ĠD');  ' D' -> 399
```

Bare letters without the prefix space exist in the vocab at `A`=34, `B`=35, `C`=36, `D`=37, but
`add_prefix_space: true` means **you cannot reach them by encoding `"A"`** — the pre-tokenizer
prepends a space (and does not double it if one is already there). So:

- The `option_mapping` / `option_tokens` duplication is a **no-op**: `"A"` and `" A"` collapse to id
  329, so the `+=` accumulation adds the same probability to `"A"` twice. Harmless for argmax, since
  every candidate is doubled equally. The comment `"Deprecated but kept for legacy reasons
  pre-tokenizer fix"` is exactly this.
- 329/378/330/399 are also the **right** ids on the merits, independent of the tokenizer quirk:
  encoding the training-time string `"... [unused0] B"` yields a tail of
  `['Ġ', '[unused0]', 'ĠB', '[SEP]']` with `ĠB` = 378. The model was trained to emit the
  space-prefixed letter at that position, so that is where the logit must be read.

**Practical consequence for a reimplementation:** if you build option ids with a tokenizer
configured with `add_prefix_space=False`, or hand-roll `vocab["A"]`, you get id 34 and the
classifier will be reading a logit the model was never trained to raise. Use
`tokenizer.encode("A", add_special_tokens=False)[0]` with the model's own tokenizer, or hardcode
329/378/330/399.

S5 resolves the same thing slightly differently (`[0]` with specials off, which is equivalent here):

```python
        token_id = tokenizer.encode(opt, add_special_tokens=False)[0]
        option_probs[opt] = probabilities[token_id].item()
```

S1/S6 do neither — the model card and blog take an unrestricted `argmax` over the whole vocabulary
and decode it:

```python
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
pred_id = outputs.logits[0, mask_idx].argmax()
answer = tokenizer.decode(pred_id)
```

S2 (the paper) **does not describe the logit-restriction mechanism at all** — I grepped the full
text for `logit`, `probabilit`, `argmax`, `verbaliz`, `candidate`, `restrict`. It discusses
verbalizers conceptually but never states that scoring is restricted to the candidate set. Not found,
not guessed.

For S4's AG News labels, I confirmed each verbose verbalizer is a single token, which is what makes
the single-mask scheme work:

```
'Business' -> [10518] ('ĠBusiness')   single_token=True
'World'    -> [3645]  ('ĠWorld')      single_token=True
'Sports'   -> [15000] ('ĠSports')     single_token=True
'Tech'     -> [18447] ('ĠTech')       single_token=True
'Sci/Tech' -> [22962, 16, 30663]      single_token=False  <- why the label map remaps it
```

---

## 5. Preceding instruction / system line

Yes, there is a leading instruction line, but its wording differs by source. No source uses a
chat-style "system" role; it is plain leading text.

- S3 `eval.py` and S2 paper (one line, then `QUESTION:` on the next):

  ```
  You will be given a question as well as a list of options. Read the question carefully and select the right answer from the list.
  ```

- S1 model card and S6 blog:

  ```
  You will be given a question and options. Select the right answer.
  ```

- S4/S5 use a two-line header form instead: `** Instructions **` (plus 4 trailing spaces) followed by
  a task sentence, e.g. `Given a short news report, determine which news category it belongs to.`

S2 §2.5.1 states training added **no** instruction line: the only templating step applied to FLAN was
prepending `[unused0]` to the `[MASK]`.

---

## 6. AG News numbers

**Only a fine-tuned number exists. There is no published zero-shot AG News number for this model.**

S1 and S2 Table 3 agree exactly (fine-tuned, accuracy):

| Model | AGNews |
|---|---|
| ModernBERT (cls head) | 95.34 |
| **ModernBERT-Large-Instruct** | **95.24** |

The zero-shot table (S1, S2 Table 2) covers **MMLU, MMLU-Pro, ADEv2, NIS, OSE** only — AG News is
not among them. ModernBERT-Large-Instruct zero-shot: MMLU 43.06, MMLU-Pro 17.16, ADEv2 53.31,
NIS 85.53, OSE 20.62, Average 43.94.

S2 confirms AG News sits in the fine-tuned section:

> "we evaluate the models on an array of widely used classification and Natural Language
> Understanding (NLU) tasks"

...and the fine-tuned section is headed "4 Full-Finetune: Can One Head Do It All?".

So **95.24 is a 2-epoch fine-tune target, not a zero-shot target.** If the goal is zero-shot AG News,
no reference number exists in any source; do not treat 95.24 as reachable zero-shot. S6 (the blog)
reports no AG News number at all.

---

## 7. `[CLS]` / `[SEP]`: left to the tokenizer

**Left entirely to the tokenizer.** No source adds them to the prompt string.

`tokenizer.json` has a `TemplateProcessing` post-processor:

```json
{"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "[SEP]", "type_id": 0}}], ...}
```

Confirmed by encoding the S3 prompt: first token `[CLS]`, last token `[SEP]`, 93 tokens total. Every
source just calls `tokenizer(text, return_tensors="pt")` with default
`add_special_tokens=True`.

This matters for `eval.py`'s `tokenizer.encode(t)[1]`: the `[1]` is skipping the auto-added `[CLS]`.

It also matters for `train.py`/`sweep_20newsgroups.py`, which find the answer position by searching
**backwards from `[SEP]`**:

```python
            sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero()
            if len(sep_positions) > 0:
                last_non_sep = sep_positions[-1].item() - 1
```

With `padding='max_length'`, `sep_positions[-1]` is the real `[SEP]` (pad is `[PAD]`=50283, not
`[SEP]`), so `-1` lands on the final answer token. That only works because the answer verbalizer is
the last content token, which is why `[unused0] {label}` must be the very end of the template.

---

## Where the sources disagree — summary

| Aspect | S3 `eval.py` + S2 paper | S1 model card + S6 blog | S4/S5 cookbook classification |
|---|---|---|---|
| Instruction | "...a question as well as a list of options. Read the question carefully and select the right answer from the list." | "...a question and options. Select the right answer." | `** Instructions **    ` + task line |
| Question placement | own line after `QUESTION:` | same line as `QUESTION: ` | under `** Content **` |
| Blank line before answer section | yes | no | no |
| Answer tail | `ANSWER:\nAnswer: [unused0] [MASK]` | `ANSWER: [unused0] [MASK]` | `** Answer **\nAnswer: [unused0] [MASK]` |
| Option form | `- A: text` | `- A: text` | `- Label: desc` (S4) / `A: label, desc` (S5) |
| Scoring | restricted to candidate ids, softmax, summed | unrestricted vocab argmax | restricted to candidate ids |

They agree on everything load-bearing: one `[MASK]`, last token, preceded by `[unused0] `, options
one per line with a `LETTER:`-or-`Label:` prefix, specials left to the tokenizer, and the compared
tokens being the space-prefixed verbalizers.

### Recommendation

Use **S3 (`eval.py`)** as the implementation reference — it is the literal code behind the published
MMLU number, and the paper's listing matches it. Keep the doubled `ANSWER:\nAnswer:` tail; it is in
both the harness and the paper, and the model was trained with the answer token in that exact
position. For AG News specifically, S4's `** Instructions **` template with the verbose
single-token labels is what produced 95.24, after fine-tuning.
