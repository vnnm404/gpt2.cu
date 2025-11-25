# import torch
# from transformers import GPT2Model, GPT2Config

# # Load GPT-2 small (124M)
# model_name = "gpt2"
# model = GPT2Model.from_pretrained(model_name)
# model.eval()  # inference mode

# # get parent directory of the script
# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))

# save_file_path = os.path.join(script_dir, "../models/gpt2-124M-weights.bin")

# # Open a file in binary write mode
# with open(save_file_path, "wb") as f:
#     # Iterate through all model parameters in order
#     for name, param in model.named_parameters():
#         # Log name and shape
#         print(f"{name}: {param.shape}")

#         # print first 4 values for verification
#         data = param.detach().cpu().float().numpy()
#         print(f"{data.flatten()[:4]}\n")

#         # a note on endianness: numpy defaults to the native endianness of the system.
#         data.tofile(f)

# print("Weights saved to gpt2-124M-weights.bin")


import torch
from transformers import GPT2Model, GPT2Config
import os

# Load GPT-2 small (124M)
model_name = "gpt2"
model = GPT2Model.from_pretrained(model_name)
model.eval()  # inference mode

# get parent directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
save_file_path = os.path.join(script_dir, "../models/gpt2-124M-weights.bin")

with open(save_file_path, "wb") as f:
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

        data = param.detach().cpu().float().numpy()

        # --- SPECIAL CASE: transpose token embeddings ---
        # GPT-2 token embedding is named: "wte.weight"
        if name == "wte.weight":
            print("Transposing wte.weight from [V, h] to [h, V]")
            data = data.T  # transpose

        # print first 4 values for verification
        print(f"{data.flatten()[:4]}\n")

        # save in system endianness
        data.tofile(f)

print("Weights saved to gpt2-124M-weights.bin")