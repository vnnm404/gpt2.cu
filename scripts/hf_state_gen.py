import struct
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from llm import GPT, GPTConfig
# ---------------- Config ----------------
OUT_FILE = "../models/gpt2_multibatch_state.bin"
BATCH_SIZE = 8
SEQ_LEN = 64
LR = 1e-4

MAGIC = 20240327
VERSION = 2

HAS_LOGITS = 1 << 0
HAS_LOSS   = 1 << 1
HAS_GRADS  = 1 << 2

# FLAGS = HAS_LOGITS | HAS_LOSS | HAS_GRADS
FLAGS = HAS_LOSS
# ----------------------------------------

device = "cuda"


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# ---- Load Tiny Shakespeare ----
with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)

# total tokens needed
tokens_per_batch = BATCH_SIZE * SEQ_LEN
max_start = len(tokens) - tokens_per_batch - 1

NUM_BATCHES = len(tokens) // tokens_per_batch
# NUM_BATCHES = 10
print(f"Num batches = {NUM_BATCHES}")


assert max_start > 0, "Tiny Shakespeare too small for chosen batch/seq size"



config = GPT2Config.from_pretrained("gpt2")

# 2. Set all dropout rates to zero (0.0)
config.embd_pdrop = 0.0
config.attn_pdrop = 0.0
config.resid_pdrop = 0.0
config.summary_pdrop = 0.0


model_config = {
    "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
    "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
    "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
    "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
}["d12"]
model = GPT(model_config)

# model = GPT2LMHeadModel.from_pretrained("gpt2", config=config).to(device)
model.train()
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

V = model.config.vocab_size
num_params = sum(p.numel() for p in model.parameters())
import tqdm
# Open output file
with open(OUT_FILE, "wb") as f:
    # ---- Header ----
    header = [0] * 256
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = NUM_BATCHES
    header[3] = BATCH_SIZE
    header[4] = SEQ_LEN
    header[5] = V
    header[6] = num_params
    header[7] = FLAGS

    f.write(struct.pack("256i", *header))

    # ---- Training loop ----
    for step in (pbar := tqdm.tqdm(range(NUM_BATCHES))):
        start = step * tokens_per_batch
        if start > max_start:
            start = start % max_start  # wrap around

        chunk = tokens[start : start + tokens_per_batch + 1]

        input_ids = chunk[:-1].view(BATCH_SIZE, SEQ_LEN).to(device)
        labels    = chunk[1:].view(BATCH_SIZE, SEQ_LEN).to(device)

        # optimizer.zero_grad(set_to_none=True)
        # logits, loss = model(input_ids, targets=labels)
        # loss.backward()
        # optimizer.step()

        # ---- Write x, y ----
        f.write(input_ids.cpu().numpy().astype("int32").tobytes())
        f.write(labels.cpu().numpy().astype("int32").tobytes())

        # ---- Write logits ----
        if FLAGS & HAS_LOGITS:
            logits = logits.detach().cpu().float()
            f.write(logits.numpy().tobytes())

        # ---- Write loss ----
        if FLAGS & HAS_LOSS:
            loss_val = 0 # loss.detach().cpu().float().item()
            pbar.set_postfix({"loss": loss_val})
            f.write(struct.pack("f", loss_val))

        # ---- Write grads ----
        if FLAGS & HAS_GRADS:
            for p in model.parameters():
                g = p.grad.detach().cpu().float().view(-1)
                f.write(g.numpy().tobytes())

print(f"Wrote {NUM_BATCHES} batches to {OUT_FILE}")
