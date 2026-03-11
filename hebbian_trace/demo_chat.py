"""Hebbian Trace Memory — GPT-2 Chat Demo.

Natural conversation with persistent memory via Hebbian trace matrices.
GPT-2 generates all responses; trace module biases logits toward stored facts.

Showcases all key mechanisms:
  - Pattern Separation 8x_k16 (interference reduction)
  - Direct Write API (template-free storage)
  - T_auto pattern completion (paraphrase resolution)
  - Hashed trace banks + best-bank scan (capacity + collision handling)
  - Reconsolidation erasure (fact updates)
  - LLM extraction via Flan-T5-small (free-text understanding)
  - Open-vocabulary: any single-BPE-token concept and entity accepted

Usage:
    python -m hebbian_trace.demo_chat
    python -m hebbian_trace.demo_chat --regex-only     # no Flan-T5
    python -m hebbian_trace.demo_chat --show-trace     # show retrieval details

Commands during chat:
    memory          Show everything the bot remembers
    reset           Clear all memories
    help            Show usage tips
    quit            Exit
"""

import argparse
import re

import torch

from .experiments.exp24_free_text import (
    AnnotatedUtterance,
    LLMExtractor,
    RegexExtractor,
    setup_model,
)
from .experiments.exp22_pattern_completion import extract_auto_pairs
from .gpt2_tasks import build_concept_vocab, ConceptEntry


# ── Open-vocabulary fact store ───────────────────────────────────────

class OpenFactStore:
    """Open-vocabulary fact store backed by Hebbian trace.

    Any single-BPE-token word can be a concept or entity.
    Predefined types from concept_vocab are loaded at init,
    then new concepts are added dynamically.
    """

    def __init__(self, tokenizer, concept_vocab: dict[str, ConceptEntry]):
        self.tokenizer = tokenizer
        self.concepts: dict[str, int] = {}
        self.entity_pools: dict[str, list[tuple[str, int]]] = {}
        self.stored: dict[str, str] = {}
        self._token_to_full: dict[tuple[str, int], str] = {}

        for type_name, entry in concept_vocab.items():
            cword = type_name
            self.concepts[cword] = entry.concept_token_id
            self.entity_pools[cword] = list(entry.entity_pool)
            for ename, eid in entry.entity_pool:
                self._token_to_full[(cword, eid)] = ename

        self._type_to_concept = {}
        for type_name in concept_vocab:
            self._type_to_concept[type_name] = type_name

    def resolve_concept(self, concept_word: str) -> int | None:
        """Get or create concept_token_id for a word."""
        if concept_word in self.concepts:
            return self.concepts[concept_word]
        toks = self.tokenizer.encode(" " + concept_word,
                                     add_special_tokens=False)
        if len(toks) != 1:
            return None
        cid = toks[0]
        self.concepts[concept_word] = cid
        self.entity_pools[concept_word] = []
        return cid

    def resolve_entity(self, concept_word: str,
                       entity_str: str) -> int | None:
        """Resolve entity to first-BPE-token ID (pointer)."""
        first_word = entity_str.split()[0]
        toks = self.tokenizer.encode(" " + first_word,
                                     add_special_tokens=False)
        if len(toks) != 1:
            return None
        eid = toks[0]
        if concept_word not in self.entity_pools:
            self.entity_pools[concept_word] = []
        pool = self.entity_pools[concept_word]
        if not any(e[1] == eid for e in pool):
            pool.append((entity_str, eid))
        self._token_to_full[(concept_word, eid)] = entity_str
        return eid

    def get_candidate_ids(self, concept_word: str) -> list[int]:
        """Get all known entity token IDs for a concept."""
        return [eid for _, eid in self.entity_pools.get(concept_word, [])]

    def token_to_display(self, concept_word: str, token_id: int) -> str:
        """Map retrieved token ID back to full entity string."""
        return self._token_to_full.get(
            (concept_word, token_id),
            self.tokenizer.decode([token_id]).strip(),
        )


# ── Extraction patterns ──────────────────────────────────────────────

_GENERIC_FACT_PATTERNS = [
    re.compile(r"my\s+(\w+)\s+is\s+(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"i\s+live\s+in\s+(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"i\s+work\s+(?:at|for)\s+(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"i(?:'m|\s+am)\s+from\s+(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"i\s+speak\s+(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"i\s+play\s+(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"i\s+drive\s+(?:a\s+)?(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"call\s+me\s+(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
]

_SHORTHAND_CONCEPT = {
    r"i\s+live\s+in\s+(.+?)[\.\!\,]?\s*$": "city",
    r"i\s+work\s+(?:at|for)\s+(.+?)[\.\!\,]?\s*$": "company",
    r"i(?:'m|\s+am)\s+from\s+(.+?)[\.\!\,]?\s*$": "country",
    r"i\s+speak\s+(.+?)[\.\!\,]?\s*$": "language",
    r"i\s+play\s+(.+?)[\.\!\,]?\s*$": "sport",
    r"i\s+drive\s+(?:a\s+)?(.+?)[\.\!\,]?\s*$": "car",
    r"call\s+me\s+(.+?)[\.\!\,]?\s*$": "name",
}
_COMPILED_SHORTHAND = {
    re.compile(k, re.IGNORECASE): v for k, v in _SHORTHAND_CONCEPT.items()
}


def generic_extract(text: str) -> list[tuple[str, str]]:
    """Extract (concept_word, entity_word) pairs from free text."""
    results = []
    m = _GENERIC_FACT_PATTERNS[0].search(text)
    if m:
        results.append((m.group(1).lower(), m.group(2)))
    for pat, concept in _COMPILED_SHORTHAND.items():
        m = pat.search(text)
        if m:
            entity = m.group(1)
            if not any(c == concept for c, _ in results):
                results.append((concept, entity))
    return results


# ── Question mapping ─────────────────────────────────────────────────

QUESTION_PATTERNS: dict[str, list[str]] = {
    "name": [r"what.+(?:my|your) name", r"who am i", r"what.+call me"],
    "city": [r"where do i live", r"what city", r"where.+live"],
    "company": [r"where do i work", r"what company", r"where.+work"],
    "color": [r"what.+fav.+color", r"what color"],
    "food": [r"what.+fav.+food", r"what food", r"what.+eat"],
    "pet": [r"what.+pet", r"what animal.+have"],
    "country": [r"what country", r"where.+from"],
    "drink": [r"what.+drink", r"what.+fav.+drink"],
    "sport": [r"what sport", r"what.+play"],
    "hobby": [r"what.+hobby", r"what.+free time"],
    "language": [r"what language", r"what.+speak"],
    "animal": [r"what.+fav.+animal"],
    "instrument": [r"what instrument", r"what.+play.+music"],
    "fruit": [r"what fruit", r"what.+fav.+fruit"],
    "flower": [r"what flower", r"what.+fav.+flower"],
    "tree": [r"what tree"],
    "metal": [r"what metal"],
    "gem": [r"what gem", r"what.+gemstone"],
    "car": [r"what car", r"what.+drive"],
    "fabric": [r"what fabric"],
    "tool": [r"what tool"],
    "day": [r"what day", r"what.+fav.+day"],
    "season": [r"what season", r"what.+fav.+season"],
    "subject": [r"what subject", r"what.+study", r"what.+fav.+subject"],
}

_COMPILED_Q_PATTERNS: dict[str, list[re.Pattern]] = {
    k: [re.compile(p, re.IGNORECASE) for p in pats]
    for k, pats in QUESTION_PATTERNS.items()
}

_GENERIC_Q = re.compile(r"what\s+is\s+my\s+(\w+)", re.IGNORECASE)


def detect_question_concept(text: str) -> str | None:
    """Map question text to concept word."""
    text_lower = text.lower().strip()
    for concept, patterns in _COMPILED_Q_PATTERNS.items():
        for pat in patterns:
            if pat.search(text_lower):
                return concept
    m = _GENERIC_Q.search(text)
    if m:
        return m.group(1).lower()
    return None


UPDATE_PATTERNS = [
    re.compile(r"actually", re.IGNORECASE),
    re.compile(r"^no[,.]", re.IGNORECASE),
    re.compile(r"not\s+.+\s+anymore", re.IGNORECASE),
    re.compile(r"changed?\s+(?:my|to)", re.IGNORECASE),
    re.compile(r"moved?\s+to", re.IGNORECASE),
    re.compile(r"now\s+(?:i|my)", re.IGNORECASE),
]


def is_update(text: str) -> bool:
    """Detect if input is a fact update (erase + rewrite)."""
    return any(p.search(text) for p in UPDATE_PATTERNS)


# ── Retrieval with top-k logits ──────────────────────────────────────

@torch.no_grad()
def retrieve_top_k(model, concept_word: str,
                   store: OpenFactStore,
                   k: int = 3) -> list[tuple[str, float, int]]:
    """Retrieve top-k predictions with confidence scores.

    Returns list of (display_name, probability, token_id).
    """
    cid = store.concepts[concept_word]
    candidate_ids = store.get_candidate_ids(concept_word)
    if not candidate_ids:
        return []

    Q = model.trace.compute_q_for_token(model._wte, cid)
    wte_weight = model._wte.weight.float()

    n_banks = model.trace.n_trace_banks
    if n_banks > 1 and model.trace._bank_traces is not None:
        cand_t = torch.tensor(candidate_ids,
                              device=model.trace.value_traces.device)
        best_logits = torch.full((len(candidate_ids),), float('-inf'),
                                 device=cand_t.device)
        for bank_id in range(n_banks):
            retrieved = model.trace.read_direct_banked(Q, bank_id)
            logits = torch.matmul(retrieved, wte_weight.T)
            cand_logits = logits[cand_t]
            best_logits = torch.max(best_logits, cand_logits)
        cand_logits = best_logits
    else:
        retrieved = model.trace.read_direct(Q)
        logits = torch.matmul(retrieved, wte_weight.T)
        cand_t = torch.tensor(candidate_ids,
                              device=model.trace.value_traces.device)
        cand_logits = logits[cand_t]

    probs = torch.softmax(cand_logits, dim=0)
    top_vals, top_ids = probs.topk(min(k, len(candidate_ids)))

    results = []
    for val, idx in zip(top_vals, top_ids):
        tid = candidate_ids[idx.item()]
        name = store.token_to_display(concept_word, tid)
        results.append((name, val.item(), tid))
    return results


# ── Natural language answer templates ────────────────────────────────

ANSWER_TEMPLATES = {
    "name": "Your name is {e}.",
    "city": "You live in {e}.",
    "company": "You work at {e}.",
    "country": "You're from {e}.",
    "color": "Your favorite color is {e}.",
    "food": "Your favorite food is {e}.",
    "pet": "Your pet is a {e}.",
    "hobby": "Your hobby is {e}.",
    "language": "You speak {e}.",
    "drink": "Your favorite drink is {e}.",
    "sport": "You play {e}.",
    "car": "You drive a {e}.",
    "age": "You are {e} years old.",
    "flower": "Your favorite flower is {e}.",
    "fruit": "Your favorite fruit is {e}.",
    "season": "Your favorite season is {e}.",
    "day": "Your favorite day is {e}.",
    "subject": "Your favorite subject is {e}.",
}

# Fill-in prompts for GPT-2 generation (concept → prompt that GPT-2 completes)
FILL_PROMPTS = {
    "name": "My name is",
    "city": "I live in",
    "company": "I work at",
    "country": "I am from",
    "color": "My favorite color is",
    "food": "My favorite food is",
    "pet": "My pet is a",
    "hobby": "My hobby is",
    "language": "I speak",
    "drink": "My favorite drink is",
    "sport": "I play",
    "car": "I drive a",
    "age": "I am",
    "flower": "My favorite flower is",
    "fruit": "My favorite fruit is",
    "season": "My favorite season is",
    "day": "My favorite day is",
    "subject": "My favorite subject is",
}

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|greetings|howdy|sup|good\s+(morning|evening|day))\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"what\s+(?:do\s+you|did\s+i)\s+(?:know|remember|tell)|"
    r"about\s+me|summarize|summary|"
    r"tell\s+me\s+(?:everything|what)\s+you\s+know",
    re.IGNORECASE,
)

# Correction patterns: "no, Andrew" / "it's Bob" / "actually Paris"
_CORRECTION_RE = [
    re.compile(r"^no[,.\s]+(?:it'?s\s+)?(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"^it'?s\s+(.+?)[\.\!\,]?\s*$", re.IGNORECASE),
    re.compile(r"^(?:i\s+meant?|actually)[,.\s]+(.+?)[\.\!\,]?\s*$",
               re.IGNORECASE),
]


def _answer(concept: str, entity: str) -> str:
    template = ANSWER_TEMPLATES.get(concept, "Your {c} is {e}.")
    return template.format(e=entity, c=concept)


def _try_correction(text: str) -> str | None:
    """Try to extract corrected entity from text like 'no, Andrew'."""
    for pat in _CORRECTION_RE:
        m = pat.match(text.strip())
        if m:
            return m.group(1).strip()
    return None


# Personal fact markers -- triggers LLM extractor for complex sentences
_PERSONAL_FACT_RE = re.compile(
    r"\b(my\s+\w+\s+is|i\s+am\s|i'm\s|i\s+live\s|i\s+work\s|"
    r"i\s+speak\s|i\s+play\s|i\s+drive\s|i\s+moved?\s|"
    r"call\s+me|i'm\s+from|i\s+study)\b",
    re.IGNORECASE,
)


def _extract_pairs(text, extractor, store):
    """Extract (concept, entity) pairs. Regex first, LLM only for complex text."""
    # 1. Try pattern-based extraction first (reliable, no false positives)
    pairs = generic_extract(text)
    if pairs:
        return pairs

    # 2. Only try LLM extractor if text looks like a personal fact statement
    #    "i have 10 apple" → no personal fact pattern → skip → GPT-2 generation
    #    "After years abroad, I settled in Paris" → "I ... in" → try LLM
    if not _PERSONAL_FACT_RE.search(text):
        return []

    utterance = AnnotatedUtterance(text=text, facts=[], tier=0)
    typed_results = extractor.extract(utterance, {
        k: ConceptEntry(
            type_name=k,
            concept_word=k,
            concept_token_id=store.concepts[k],
            fact_template="",
            question_template="",
            entity_pool=store.entity_pools[k],
        )
        for k in store.concepts
        if k in store.entity_pools and store.entity_pools[k]
    })
    seen: set[str] = set()
    for fact in typed_results:
        key = fact.entity.strip().lower()
        if key not in seen:
            seen.add(key)
            pairs.append((fact.type_name, fact.entity))
    return pairs


# ── GPT-2 generation ────────────────────────────────────────────────

@torch.no_grad()
def _gpt2_generate(model, tokenizer, prompt: str,
                   max_new: int = 30, use_trace: bool = True) -> str:
    """Generate text continuation with GPT-2 (± trace augmentation)."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    model.set_trace_mode(use=use_trace, update=False)

    # Stop at newline or EOS
    stop = []
    if tokenizer.eos_token_id is not None:
        stop.append(tokenizer.eos_token_id)
    nl_ids = tokenizer.encode('\n')
    if nl_ids:
        stop.append(nl_ids[0])

    gen_ids = model.generate(
        input_ids, max_new_tokens=max_new, stop_token_ids=stop or None)

    text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    return text.strip()


def _generate_memory_answer(model, tokenizer, concept: str,
                            use_trace: bool = True) -> str:
    """Generate answer for a memory concept using fill-in prompt + GPT-2."""
    prompt = FILL_PROMPTS.get(concept, f"My {concept} is")
    return _gpt2_generate(model, tokenizer, prompt,
                          max_new=5, use_trace=use_trace)


# ── Chat loop ────────────────────────────────────────────────────────

def interactive_chat(model, tokenizer, store, extractor, banks, show_trace):
    """Natural chat loop with Hebbian trace memory."""
    model._tokenizer = tokenizer
    tauto = getattr(model, '_tauto_loaded', False)
    n_writes = 0
    last_concept = None  # Track last asked concept for corrections

    print()
    print("=" * 56)
    print("  Hebbian Trace Memory -- GPT-2 Chat Demo")
    print("=" * 56)
    print()
    print("  Model:  GPT-2 Small (124M params, frozen)")
    print("  Memory: Hebbian trace matrices (no backprop)")
    mechanisms = ["pattern_separation"]
    if banks > 0:
        mechanisms.append(f"banks={banks}")
    if tauto:
        mechanisms.append("T_auto")
    mechanisms.append("erasure")
    print(f"  Active: {', '.join(mechanisms)}")
    print()
    print("Bot: Hi! I'm GPT-2 with a Hebbian trace memory module.")
    print("     Tell me about yourself -- I'll remember everything.")
    print("     (Type 'memory' to see what I know, 'quit' to exit)")
    print()

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!\n")
            break

        if not line:
            continue
        cmd = line.lower()

        # ── Commands ──
        if cmd in ("quit", "exit", "q"):
            print("Bot: Goodbye!\n")
            break

        if cmd in ("memory", "status"):
            if store.stored:
                print("Bot: Here's what I remember about you:")
                for c, e in sorted(store.stored.items()):
                    print(f"       {_answer(c, e)}")
                print(f"     ({len(store.stored)} facts in trace memory)")
            else:
                print("Bot: I don't know anything yet -- tell me!")
            print()
            continue

        if cmd == "reset":
            model.reset_traces()
            store.stored.clear()
            n_writes = 0
            last_concept = None
            print("Bot: Memory cleared! Let's start fresh.\n")
            continue

        if cmd == "help":
            print("Bot: Just talk to me naturally!")
            print("       My name is Andrew")
            print("       I live in Paris and work at Google")
            print("       What is my name?")
            print("       no, Bob              (correct last answer)")
            print("       Actually, I moved to London")
            print("       What do you know about me?")
            print("       2 + 2?               (general question)")
            print("     Commands: memory, reset, help, quit\n")
            continue

        is_greet = bool(_GREETING_RE.search(line))

        # ── Summary request ──
        if _SUMMARY_RE.search(line):
            if store.stored:
                print("Bot: Here's what I know about you:")
                for c, e in sorted(store.stored.items()):
                    print(f"       {_answer(c, e)}")
            else:
                print("Bot: I don't know anything about you yet!")
            print()
            continue

        # ── Correction ("no, Andrew" / "it's Bob") ──
        if last_concept and not is_greet:
            corrected = _try_correction(line)
            if corrected:
                cid = store.resolve_concept(last_concept)
                eid = store.resolve_entity(last_concept, corrected) \
                    if cid else None
                if cid and eid:
                    old = store.stored.get(last_concept)
                    model.set_erase_mode(True, erase_lr=5.0)
                    model.write_fact_direct(cid, eid)
                    model.set_erase_mode(False)
                    store.stored[last_concept] = corrected
                    n_writes += 1
                    if old:
                        print(f"Bot: Updated! "
                              f"{last_concept}: {old} -> {corrected}.")
                    else:
                        print(f"Bot: Got it! I'll remember your "
                              f"{last_concept} is {corrected}.")
                    last_concept = None
                    print()
                    continue

        # ── Question ──
        if line.rstrip().endswith("?") and not is_greet:
            concept = detect_question_concept(line)

            # Memory question with stored fact
            if concept:
                cid = store.resolve_concept(concept)
                cand_ids = store.get_candidate_ids(concept) \
                    if cid else []

                if cand_ids:
                    last_concept = concept
                    top_k = retrieve_top_k(model, concept, store, k=5)
                    if top_k:
                        best_name, best_prob, _ = top_k[0]
                        answer = _answer(concept, best_name)
                        if best_prob < 0.5:
                            print(f"Bot: I'm not very sure, but... "
                                  f"{answer}")
                        else:
                            print(f"Bot: {answer}")

                        if show_trace:
                            # Show trace retrieval
                            for name, prob, _ in top_k[:3]:
                                bar = "#" * int(prob * 20)
                                print(f"       {name:15s} "
                                      f"{prob:5.1%} {bar}")
                            # Show what GPT-2 generates WITHOUT trace
                            baseline = _generate_memory_answer(
                                model, tokenizer, concept, use_trace=False)
                            print(f"     [GPT-2 without trace: "
                                  f'"{baseline}"]')
                        print()
                        continue
                    else:
                        print(f"Bot: I can't recall your {concept}.\n")
                        continue

                elif concept in store.concepts:
                    last_concept = concept
                    print(f"Bot: You haven't told me your "
                          f"{concept} yet.\n")
                    continue

            # General question -- GPT-2 free generation
            last_concept = None
            response = _gpt2_generate(
                model, tokenizer, line, max_new=30, use_trace=True)
            if response and response.lower() != line.lower():
                print(f"Bot: {response}")
            else:
                # GPT-2 didn't generate anything useful
                response = _gpt2_generate(
                    model, tokenizer, line.rstrip("?") + " =",
                    max_new=10, use_trace=True)
                if response:
                    print(f"Bot: {response}")
                else:
                    print("Bot: I'm not sure about that.")
            print()
            continue

        # ── Fact / Update ──
        updating = is_update(line)
        pairs = _extract_pairs(line, extractor, store)

        # Greeting without facts
        if is_greet and not pairs:
            name = store.stored.get("name")
            if name:
                print(f"Bot: Hey, {name}! What's new?")
            elif store.stored:
                print("Bot: Hey! Good to see you again.")
            else:
                print("Bot: Hello! Tell me about yourself "
                      "-- I'll remember everything.")
            print()
            continue

        # No facts extracted -- use GPT-2 free generation
        if not pairs:
            last_concept = None
            response = _gpt2_generate(
                model, tokenizer, line, max_new=30, use_trace=True)
            if response and response.lower() != line.lower():
                print(f"Bot: {response}")
            else:
                print("Bot: " + _gpt2_generate(
                    model, tokenizer, line + ".",
                    max_new=20, use_trace=True))
            print()
            continue

        if updating:
            model.set_erase_mode(True, erase_lr=5.0)

        msgs = []
        for concept_word, entity_word in pairs:
            cid = store.resolve_concept(concept_word)
            if cid is None:
                continue
            eid = store.resolve_entity(concept_word, entity_word)
            if eid is None:
                continue

            old = store.stored.get(concept_word)
            model.write_fact_direct(cid, eid)
            store.stored[concept_word] = entity_word
            n_writes += 1

            if updating and old and old.lower() != entity_word.lower():
                msgs.append(("update", concept_word, old, entity_word))
            else:
                msgs.append(("store", concept_word, None, entity_word))

        if updating:
            model.set_erase_mode(False)

        if not msgs:
            print("Bot: Couldn't store that -- "
                  "entity might not be a single BPE token.\n")
            continue

        last_concept = None

        # Format response
        has_update = any(m[0] == "update" for m in msgs)
        if has_update and is_greet:
            name = next((e for _, c, _, e in msgs if c == "name"), None)
            prefix = "Hey! " if not name else f"Hey, {name}! "
        elif is_greet:
            name = next((e for _, c, _, e in msgs if c == "name"), None)
            prefix = f"Nice to meet you, {name}! " if name \
                else "Hello! "
        elif has_update:
            prefix = "Updated! "
        else:
            prefix = "Got it! "

        if len(msgs) == 1:
            kind, concept, old, entity = msgs[0]
            if kind == "update":
                print(f"Bot: {prefix}{concept}: {old} -> {entity}.")
            else:
                print(f"Bot: {prefix}I'll remember your "
                      f"{concept} is {entity}.")
        else:
            print(f"Bot: {prefix}Noted:")
            for kind, concept, old, entity in msgs:
                if kind == "update":
                    print(f"       {concept}: {old} -> {entity}")
                else:
                    print(f"       {concept} = {entity}")

        if show_trace:
            print(f"     [{n_writes} facts in trace]")

        print()


# ── Entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chat demo with Hebbian trace memory")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--no-tauto", action="store_true")
    parser.add_argument("--regex-only", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--show-trace", action="store_true",
                        help="Show trace retrieval details")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    print("Loading GPT-2 + Hebbian trace module...")
    model, tokenizer = setup_model(
        alpha=args.alpha, use_ps=True, device=device)

    concept_vocab = build_concept_vocab(tokenizer)
    store = OpenFactStore(tokenizer, concept_vocab)

    if args.banks > 0:
        model.set_bank_mode(args.banks)

    if not args.no_tauto:
        auto_pairs = extract_auto_pairs(tokenizer)
        pair_tuples = [(p.variant_id, p.concept_id) for p in auto_pairs]
        model.write_auto_pairs(pair_tuples)
        model.set_auto_mode(True, completion_alpha=1.0)
        model._tauto_loaded = True
    else:
        model._tauto_loaded = False

    if args.regex_only:
        extractor = RegexExtractor()
    else:
        extractor = LLMExtractor(device=device)

    model.set_erase_mode(False)
    print("Ready!")

    interactive_chat(model, tokenizer, store, extractor,
                     args.banks, args.show_trace)


if __name__ == "__main__":
    main()
