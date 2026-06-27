import argparse
import os
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Train grammatical fluency classifier.")
parser.add_argument("--epochs",       type=int,   default=3,               help="Epochs (default: 3)")
parser.add_argument("--samples",      type=int,   default=20000,           help="Sentence pairs (default: 20000)")
parser.add_argument("--batch-size",   type=int,   default=32,              help="Batch size (default: 32)")
parser.add_argument("--lr",           type=float, default=2e-5,            help="Learning rate (default: 2e-5 for fine-tuning BERT)")
parser.add_argument("--resume",       action="store_true",                 help="Resume from checkpoint")
parser.add_argument("--checkpoint",   type=str,   default="checkpoint.pt", help="Checkpoint path (default: checkpoint.pt)")
parser.add_argument("--max-sent-len", type=int,   default=30,              help="Max words per sentence (default: 30)")
parser.add_argument("--max-tokens",   type=int,   default=64,              help="Max subword tokens for BERT (default: 64)")
args = parser.parse_args()

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = args.batch_size
EPOCHS       = args.epochs
LR           = args.lr
MAX_SENT_LEN = args.max_sent_len
MAX_TOKENS   = args.max_tokens
CHECKPOINT   = args.checkpoint
MODEL_NAME   = "prajjwal1/bert-mini"   # 11MB, used bert_base before

print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Samples: {args.samples} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"Encoder: {MODEL_NAME} (fine-tuned end-to-end)")


# bert-mini ships a broken tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
model     = model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
loss_fn   = nn.BCEWithLogitsLoss()   # model now outputs raw logits, not sigmoid


start_epoch = 0

if args.resume and os.path.exists(CHECKPOINT):
    print(f"Resuming from {CHECKPOINT} ...")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"]
    print(f"  => Resumed after epoch {start_epoch} (loss was {ckpt.get('loss', 0):.4f})")
elif args.resume:
    print(f"No checkpoint found at '{CHECKPOINT}', starting fresh.")


# budget sentence splitting
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def split_sentences(text: str):
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for sent in _SENT_SPLIT.split(para):
            sent = sent.strip()
            if sent:
                yield sent


# mess stuff up somewhat predictaby
ARTICLES     = {"a", "an", "the"}
CONJUNCTIONS = {"and", "but", "or", "nor", "so", "yet", "for"}

SV_AGREEMENT = {
    "is": "are", "are": "is",
    "was": "were", "were": "was",
    "has": "have", "have": "has",
    "does": "do", "do": "does",
}

def _mangle_tense(token: str):
    t = token.lower()
    if t.endswith("ing") and len(t) > 5:
        return t[:-3]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and len(t) > 4 and t not in SV_AGREEMENT:
        return t[:-1]
    return None


# pronouns used as subjects (or vice versa)
PRONOUN_SWAP = {
    "i": "me", "me": "i",
    "he": "him", "him": "he",
    "she": "her", "her": "she",
    "we": "us", "us": "we",
    "they": "them", "them": "they",
}

# 
BARE_PARTICIPLES = {
    "seen", "done", "gone", "been", "taken", "given", "come",
    "run", "known", "grown", "shown", "spoken", "written",
    "broken", "chosen", "fallen", "forgotten", "gotten",
    "hidden", "risen", "stolen", "thrown", "worn", "woken",
}


def corrupt(sentence: str) -> str:
    tokens = sentence.split()
    if len(tokens) < 4:
        return sentence

    pool    = list(range(7))   # 7 corruption types
    random.shuffle(pool)
    applied = 0
    target  = random.randint(1, 2)

    for cid in pool:
        if applied >= target:
            break

        if cid == 0:  # S-V agreement
            positions = [i for i, t in enumerate(tokens) if t.lower() in SV_AGREEMENT]
            if positions:
                i = random.choice(positions)
                tokens[i] = SV_AGREEMENT[tokens[i].lower()]
                applied += 1

        elif cid == 1:  # tense/morphology
            candidates = [(i, _mangle_tense(t)) for i, t in enumerate(tokens) if _mangle_tense(t)]
            if candidates:
                i, mangled = random.choice(candidates)
                tokens[i] = mangled
                applied += 1

        elif cid == 2:  # article duplication / spurious insertion
            art_pos = [i for i, t in enumerate(tokens) if t.lower() in ARTICLES]
            if art_pos and random.random() < 0.5:
                i = random.choice(art_pos)
                tokens.insert(i, tokens[i])          # duplicate: "the the"
                applied += 1
            else:
                non_art = [i for i, t in enumerate(tokens) if t.lower() not in ARTICLES]
                if non_art:
                    tokens.insert(random.choice(non_art), random.choice(["a", "an", "the"]))
                    applied += 1

        elif cid == 3:  # window shuffle
            if len(tokens) >= 6:
                start    = random.randint(1, max(1, len(tokens) - 5))
                size     = random.randint(4, min(6, len(tokens) - start))
                window   = tokens[start:start + size]
                shuffled = window[:]
                random.shuffle(shuffled)
                if shuffled != window:
                    tokens[start:start + size] = shuffled
                    applied += 1

        elif cid == 4:  # conjunction stranding
            conj_pos = [i for i, t in enumerate(tokens) if t.lower() in CONJUNCTIONS]
            if conj_pos:
                i    = random.choice(conj_pos)
                conj = tokens.pop(i)
                tokens.insert(0 if random.random() < 0.5 else len(tokens), conj)
                applied += 1

        elif cid == 5:  # article drop
            art_pos = [i for i, t in enumerate(tokens) if t.lower() in ARTICLES]
            if art_pos:
                tokens.pop(random.choice(art_pos))
                applied += 1

        elif cid == 6:  # pronoun case / bare participle
            if tokens[0].lower() in PRONOUN_SWAP and random.random() < 0.5:
                tokens[0] = PRONOUN_SWAP[tokens[0].lower()]
                applied += 1
            else:
                aux = {"have", "has", "had"}
                aux_pos = [
                    i for i, t in enumerate(tokens)
                    if t.lower() in aux
                    and i + 1 < len(tokens)
                    and tokens[i + 1].lower() in BARE_PARTICIPLES
                ]
                if aux_pos:
                    tokens.pop(random.choice(aux_pos))
                    applied += 1

    return " ".join(tokens)


# ----------------------------
# Hard examples, courtesy ChatGPT.
# ----------------------------
HARD_EXAMPLES: list[tuple[str, float, float]] = [
    # --- subtle 3sg-s agreement errors (label=0, fluent pair label=1) ---
    ("The committee approves the new budget every year.",           1.0, 20.0),
    ("The committee approve the new budget every year.",            0.0, 20.0),
    ("The government introduces new legislation each session.",     1.0, 20.0),
    ("The government introduce new legislation each session.",      0.0, 20.0),
    ("The researchers publishes their findings annually.",          0.0, 20.0),
    ("The researchers publish their findings annually.",            1.0, 20.0),
    ("The team wins most of its home games.",                       1.0, 20.0),
    ("The team win most of its home games.",                        0.0, 20.0),
    ("The company reports record profits this quarter.",            1.0, 20.0),
    ("The company report record profits this quarter.",             0.0, 20.0),
    ("The board approves the decision by a majority vote.",         1.0, 20.0),
    ("The board approve the decision by a majority vote.",          0.0, 20.0),
    ("The audience applauds after every performance.",              1.0, 20.0),
    ("The audience applaud after every performance.",               0.0, 20.0),  # both valid actually, but weight the singular form
    ("The press reports on the story every day.",                   1.0, 20.0),
    ("The press report on the story every day.",                    0.0, 20.0),

    # --- trailing conjunction errors ---
    ("She finished her homework and went to bed.",                  1.0, 20.0),
    ("She finished her homework and went to bed and.",              0.0, 20.0),
    ("I said hello but he ignored me.",                             1.0, 20.0),
    ("I said hello but.",                                           0.0, 20.0),
    ("He wanted to leave but stayed anyway.",                       1.0, 20.0),
    ("He wanted to leave but stayed anyway but.",                   0.0, 20.0),
    ("They tried hard yet still failed.",                           1.0, 20.0),
    ("They tried hard yet still failed yet.",                       0.0, 20.0),
    ("We could stay or we could go.",                               1.0, 20.0),
    ("We could stay or we could go or.",                            0.0, 20.0),

    # --- valid inverted / non-canonical syntax (should be fluent) ---
    ("To the store he went.",                                       1.0, 20.0),
    ("Down the hill rolled the barrel.",                            1.0, 20.0),
    ("Never had she seen such a sight.",                            1.0, 20.0),
    ("Rarely does he make such mistakes.",                          1.0, 20.0),
    ("Only then did they realise the truth.",                       1.0, 20.0),
    ("Into the room walked a tall stranger.",                       1.0, 20.0),
    ("Out of the forest came a strange sound.",                     1.0, 20.0),
    ("Here lies a great warrior.",                                  1.0, 20.0),
    ("There goes the last train of the night.",                     1.0, 20.0),
    ("Up the stairs ran the children.",                             1.0, 20.0),
    ("Slowly and carefully, she opened the envelope.",              1.0, 20.0),
    ("Not until morning did the storm finally pass.",               1.0, 20.0),
    ("Among the guests was a famous author.",                       1.0, 20.0),
    ("On the table sat a small wooden box.",                        1.0, 20.0),
    ("With great power comes great responsibility.",                1.0, 20.0),

    # --- article drop errors ("went to store", "at hospital") ---
    ("She went to the store to buy some milk.",                     1.0, 20.0),
    ("She went to store to buy some milk.",                         0.0, 20.0),
    ("He was admitted to the hospital last night.",                 1.0, 20.0),
    ("He was admitted to hospital last night.",                     0.0, 20.0),
    ("They arrived at the airport just in time.",                   1.0, 20.0),
    ("They arrived at airport just in time.",                       0.0, 20.0),
    ("She sat down at the table and opened her book.",              1.0, 20.0),
    ("She sat down at table and opened her book.",                  0.0, 20.0),
    ("He put the keys on the counter before leaving.",              1.0, 20.0),
    ("He put keys on counter before leaving.",                      0.0, 20.0),
    ("We met at the park near the old church.",                     1.0, 20.0),
    ("We met at park near old church.",                             0.0, 20.0),
    ("She is studying at the university downtown.",                 1.0, 20.0),
    ("She is studying at university downtown.",                     0.0, 20.0),

    # --- pronoun case errors ("him and me went", "her did it") ---
    ("She and I went to the market together.",                      1.0, 20.0),
    ("Her and me went to the market together.",                     0.0, 20.0),
    ("He and I decided to leave early.",                            1.0, 20.0),
    ("Him and me decided to leave early.",                          0.0, 20.0),
    ("They invited her and me to the party.",                       1.0, 20.0),
    ("They invited she and I to the party.",                        0.0, 20.0),
    ("We told him the truth.",                                      1.0, 20.0),
    ("Us told he the truth.",                                       0.0, 20.0),
    ("She gave them the report on Friday.",                         1.0, 20.0),
    ("Her gave they the report on Friday.",                         0.0, 20.0),

    # --- bare past participle ("I seen", "he done", "she gone") ---
    ("I have seen this movie before.",                              1.0, 20.0),
    ("I seen this movie before.",                                   0.0, 20.0),
    ("She has gone to the library already.",                        1.0, 20.0),
    ("She gone to the library already.",                            0.0, 20.0),
    ("He has done all the work himself.",                           1.0, 20.0),
    ("He done all the work himself.",                               0.0, 20.0),
    ("They have taken the last train home.",                        1.0, 20.0),
    ("They taken the last train home.",                             0.0, 20.0),
    ("We have spoken about this before.",                           1.0, 20.0),
    ("We spoken about this before.",                                0.0, 20.0),
    ("She has written three novels so far.",                        1.0, 20.0),
    ("She written three novels so far.",                            0.0, 20.0),
    ("He has broken the record twice.",                             1.0, 20.0),
    ("He broken the record twice.",                                 0.0, 20.0),

    # --- quantifier-of-plural agreement ("a number of X were", not "was") ---
    # The head noun is singular ("number", "group") but the verb must agree
    # with the plural complement, making these a common native-speaker trap.
    ("A number of students were absent from class today.",          1.0, 20.0),
    ("A number of students was absent from class today.",           0.0, 20.0),
    ("A group of researchers were studying the phenomenon.",        1.0, 20.0),
    ("A group of researchers was studying the phenomenon.",         0.0, 20.0),
    ("A series of experiments were conducted over two years.",      1.0, 20.0),
    ("A series of experiments was conducted over two years.",       0.0, 20.0),
    ("A pair of scientists were awarded the prize.",                1.0, 20.0),
    ("A pair of scientists was awarded the prize.",                 0.0, 20.0),
    ("A number of issues were raised during the meeting.",          1.0, 20.0),
    ("A number of issues was raised during the meeting.",           0.0, 20.0),
    ("A range of options were available to the committee.",         1.0, 20.0),
    ("A range of options was available to the committee.",          0.0, 20.0),
    ("A collection of artefacts were found at the dig site.",       1.0, 20.0),
    ("A collection of artefacts was found at the dig site.",        0.0, 20.0),
    ("A variety of species were observed in the study area.",       1.0, 20.0),
    ("A variety of species was observed in the study area.",        0.0, 20.0),
]



class FluencyDataset(Dataset):
    def __init__(self, max_samples: int):
        print(f"Loading C4 (up to {max_samples} sentence pairs)...")
        wiki = load_dataset("allenai/c4", "en", split="train", streaming=True)

        self.samples = []   # (sentence, label)
        self.weights = []   # sampling weight
        count = 0

        for item in tqdm(wiki, desc="Scanning articles"):
            for s in split_sentences(item["text"]):
                words = s.split()
                if len(words) < 6 or len(words) > MAX_SENT_LEN:
                    continue

                self.samples.append((s, 1.0))
                self.weights.append(1.0)
                self.samples.append((corrupt(s), 0.0))
                self.weights.append(1.0)

                count += 1
                if count >= max_samples:
                    break
            if count >= max_samples:
                break

        #hard examples with boosted weights
        print(f"{len(HARD_EXAMPLES)} hard examples...")
        for sentence, label, weight in HARD_EXAMPLES:
            self.samples.append((sentence, label))
            self.weights.append(weight)

        print(f"Dataset: {len(self.samples)} total samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def collate(batch):
    sentences, labels = zip(*batch)

    encoded = tokenizer(
        list(sentences),
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )

    input_ids      = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    labels_t       = torch.tensor(labels, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    return input_ids, attention_mask, labels_t



dataset  = FluencyDataset(max_samples=args.samples)
sampler  = WeightedRandomSampler(
    weights     = dataset.weights,
    num_samples = len(dataset),
    replacement = True,
)
loader   = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                      collate_fn=collate, num_workers=0)

total_steps   = EPOCHS * len(loader)
warmup_steps  = total_steps // 10

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Fast-forward scheduler if resuming
if start_epoch > 0:
    steps_done = start_epoch * len(loader)
    for _ in range(steps_done):
        scheduler.step()



# mainloop
model.train()

for epoch in range(start_epoch, start_epoch + EPOCHS):
    total_loss = 0
    correct    = 0

    for input_ids, attention_mask, labels in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # standard for BERT fine-tuning apparently
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = (logits.sigmoid() > 0.5)
        correct    += (preds == labels.bool()).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(dataset)
    print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f} (avg {avg_loss:.4f}) | Accuracy: {accuracy:.2%}")

    torch.save({
        "epoch":     epoch + 1,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss":      total_loss,
    }, CHECKPOINT)
    print(f"Checkpoint saved to {CHECKPOINT}")



model.save_pretrained("fluency_model")
tokenizer.save_pretrained("fluency_model")
print("Saved to ./fluency_model/")