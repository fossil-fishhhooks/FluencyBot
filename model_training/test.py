import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR   = "fluency_model"
CHECKPOINT  = "checkpoint.pt"
MODEL_NAME  = "prajjwal1/bert-mini"   # must match


if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print(f"Loading from ./{MODEL_DIR}/ ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
elif os.path.exists(CHECKPOINT):
    print(f"./{MODEL_DIR}/ not found — loading weights from {CHECKPOINT} ...")
    # bert-mini ships a broken tokenizer. idk man
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    ckpt      = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    epoch     = ckpt.get("epoch", "?")
    print(f"  => Loaded checkpoint (epoch {epoch})")
else:
    raise FileNotFoundError(
        f"No model found. Expected either ./{MODEL_DIR}/ or {CHECKPOINT}.\n"
        f"Run train.py first, or resume an interrupted run with --resume."
    )

model = model.to(DEVICE)
model.eval()
print(f"Ready on {DEVICE}.\n")



def score(sentence: str) -> float:
    """Return fluency score: 1.0 = fluent, 0.0 = broken."""
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    input_ids      = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    return logits.sigmoid().item()


def evaluate(sentences: list):
    print(f"{'Score':>6}  {'Label':<12}  Sentence")
    print("-" * 85)
    for label, sent in sentences:
        s      = score(sent)
        marker = "OK" if s >= 0.5 else "X "
        print(f"  {s:.3f}  {marker} {label:<10}  {sent}")
    print()


# ----------------------------
# Test battery
# ----------------------------
evaluate([
    # Fluent
    ("fluent",       "The researchers published their findings in a peer-reviewed journal."),
    ("fluent",       "She has been working at the company for three years."),
    ("fluent",       "The old bridge was demolished and replaced with a modern structure."),
    ("fluent",       "He quickly ran to the store before it closed."),
    ("fluent",       "Both teams played well, but only one could win."),

    # S-V agreement
    ("s-v agree",    "The researchers publishes their findings in a peer-reviewed journal."),
    ("s-v agree",    "She have been working at the company for three years."),
    ("s-v agree",    "The dogs is barking loudly outside the window."),

    # Morphology / tense
    ("morphology",   "He quick ran to the store before it close."),
    ("morphology",   "They was walk to school when it start to rain."),

    # Word order
    ("word order",   "Published their the researchers findings in journal a peer-reviewed."),
    ("word order",   "Quickly the to store ran he before closed it."),

    # Article errors
    ("articles",     "She saw the the cat sitting on a mat."),
    ("articles",     "He went to a university on the Tuesday the morning."),

    # Conjunction stranding
    ("conjunction",  "And the old bridge was demolished replaced with a modern structure."),
    ("conjunction",  "Both teams played well but only one could win but."),
])


print("Enter sentences to score (blank line to quit):\n")
while True:
    try:
        line = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        break
    print(f"  Score: {score(line):.4f}\n")