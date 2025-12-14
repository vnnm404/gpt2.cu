import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
import numpy as np
import os

def load_gpt2_weights_from_bin(model, bin_path):
    model.eval()
    count = 0
    with open(bin_path, "rb") as f:
        for name, param in model.named_parameters():
            # Number of elements to read
            numel = param.numel()
            count += numel
            # print(name, count)

            # Read raw bytes
            data = np.fromfile(f, dtype=np.float32, count=numel)
            if data.size != numel:
                raise RuntimeError(f"Failed to read enough data for {name}")

            # Reshape

            # --- SPECIAL CASE: transpose token embeddings ---
            if  "wte.weight" in name:
                data = data.reshape(param.shape[::-1])
                # print("Transposing wte.weight from [h, V] to [V, h]")
                data = data.T
                print(data[:5][:5])
            else:
                data = data.reshape(param.shape)


            # Load into parameter
            param.data.copy_(torch.from_numpy(data))

            # Debug print
            # print(f"{name}: loaded {param.shape}")
            # print(f"{param.data.flatten()[:4]}\n")

    return model


def are_gpt2_state_dicts_equal(state_dict_1, state_dict_2):
    """
    Compares two PyTorch state_dicts to check if they have identical keys and tensor values.

    Args:
        state_dict_1 (dict): The first state dictionary.
        state_dict_2 (dict): The second state dictionary.

    Returns:
        bool: True if the state dicts are identical, False otherwise.
    """
    # Check if the number of keys is the same
    if len(state_dict_1) != len(state_dict_2):
        print(f"Length mismatch: {len(state_dict_1)} vs {len(state_dict_2)}")
        return False

    # Iterate through keys and compare corresponding tensors
    for key in state_dict_1:
        if key not in state_dict_2:
            print(f"Key '{key}' not found in the second state_dict.")
            return False

        tensor_1 = state_dict_1[key]
        tensor_2 = state_dict_2[key]

        # Use torch.equal for exact tensor comparison
        if not torch.equal(tensor_1, tensor_2):
            print(f"Mismatch found at tensor for key: '{key}'")
            # Optional: print specific values/shapes for debugging
            print(f"Shape 1: {tensor_1.shape}, Shape 2: {tensor_2.shape}")
            print(f"First 5 elements of tensor 1: {tensor_1.flatten()[:5]}")
            print(f"First 5 elements of tensor 2: {tensor_2.flatten()[:5]}")
            return False

    print("All state_dict tensors match perfectly!")
    return True


BATCH_SIZE = 8
SEQ_LEN = 64


device = "cuda"


model_name = "gpt2"
model = load_gpt2_weights_from_bin(GPT2LMHeadModel.from_pretrained(model_name), "../models/gpt2-124M-weights.bin")
new_model = load_gpt2_weights_from_bin(GPT2LMHeadModel.from_pretrained(model_name), "../models/checkpoint.bin")
print(are_gpt2_state_dicts_equal(GPT2LMHeadModel.from_pretrained(model_name).state_dict(), model.state_dict()))


model.to(device)
new_model.to(device)


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
NUM_BATCHES = 10
print(f"Num batches = {NUM_BATCHES}")


assert max_start > 0, "Tiny Shakespeare too small for chosen batch/seq size"

model.train()
new_model.train()

for step in ((range(NUM_BATCHES))):
    start = step * tokens_per_batch
    if start > max_start:
        start = start % max_start  # wrap around

    chunk = tokens[start : start + tokens_per_batch + 1]

    input_ids = chunk[:-1].view(BATCH_SIZE, SEQ_LEN).to(device)
    labels    = chunk[1:].view(BATCH_SIZE, SEQ_LEN).to(device)

    out = model(input_ids, labels=labels)
    new_out = new_model(input_ids, labels=labels)

    print(out.loss, new_out.loss)