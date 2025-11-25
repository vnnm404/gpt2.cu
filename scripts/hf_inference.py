import torch
from transformers import GPT2Model, GPT2Tokenizer

# Load GPT-2 small (without LM head)
model_name = "gpt2"
model = GPT2Model.from_pretrained(model_name)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # set pad token

# Input prompt
input_text = "The capital of France is"
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

# Number of tokens to generate
num_tokens_to_generate = 10

# Prepare hidden input_ids for iterative generation
generated_ids = input_ids.clone()

for _ in range(num_tokens_to_generate):
    with torch.no_grad():
        # Get hidden states from GPT-2 transformer
        outputs = model(generated_ids)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # Get GPT-2 token embeddings
        embedding_weights = model.wte.weight  # [vocab_size, hidden_size]

        # Compute logits for the last token only
        logits = torch.matmul(hidden_states[:, -1, :], embedding_weights.T)  # [batch, vocab_size]

        # Pick the token with highest probability (greedy)
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)  # shape [batch, 1]

        # Append the predicted token to generated_ids
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

# Decode generated text
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Prompt: ", input_text)
print("Generated Text: ", generated_text)


# import torch
# from transformers import GPT2Model, GPT2Tokenizer

# # Load GPT-2 small (without LM head)
# model_name = "gpt2"
# model = GPT2Model.from_pretrained(model_name)
# model.eval()

# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # set pad token

# # Input prompt
# input_text = "The capital of France is"
# inputs = tokenizer(input_text, return_tensors="pt")
# input_ids = inputs["input_ids"]  # [1, seq_len]

# # ---- Get embedding output (token + position embeddings) ----
# with torch.no_grad():
#     outputs = model(input_ids, output_hidden_states=True)
#     embedding_output = outputs.hidden_states[0]    # [1, seq_len, hidden_size]
#     first_token_embedding_after_pos = embedding_output[0, 1]  # [hidden_size]

# # print("Embedding vector shape:", first_token_embedding_after_pos.shape)
# # print("Embedding values:\n", first_token_embedding_after_pos)

# import torch
# from transformers import GPT2Model, GPT2Tokenizer

# # Load GPT-2 small (without LM head)
# model_name = "gpt2"
# model = GPT2Model.from_pretrained(model_name)
# model.eval()

# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# # Input prompt
# input_text = "The capital of France is"
# inputs = tokenizer(input_text, return_tensors="pt")
# input_ids = inputs["input_ids"]


# # -------- Hook to capture ln_1 output --------
# ln1_activations = {}

# def ln1_hook(module, input, output):
#     # output is the tensor after ln_1: shape [batch, seq, hidden]
#     ln1_activations["value"] = output.detach()

# # Register hook on the first transformer block (block 0)
# hook = model.h[0].ln_1.register_forward_hook(ln1_hook)


# # -------- Run model --------
# with torch.no_grad():
#     _ = model(input_ids)

# # Remove hook
# hook.remove()

# # Extract the activations
# ln1_output = ln1_activations["value"]        # [1, seq_len, hidden_size]
# first_token_after_ln1 = ln1_output[0, 1]     # token index 1, vector size 768

# print("Shape after ln_1:", ln1_output.shape)
# print("First token (after ln_1) vector shape:", first_token_after_ln1.shape)
# print(first_token_after_ln1)

# import torch
# from transformers import GPT2Model, GPT2Tokenizer

# # Load GPT-2 small (without LM head)
# model_name = "gpt2"
# model = GPT2Model.from_pretrained(model_name)
# model.eval()

# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# # Input prompt
# input_text = "The capital of France is"
# inputs = tokenizer(input_text, return_tensors="pt")
# input_ids = inputs["input_ids"]

# # -------- Hook to capture c_attn output (QKV combined) --------
# c_attn_activations = {}

# def c_attn_hook(module, input, output):
#     # output shape: [batch, seq_len, 3 * hidden_size]
#     c_attn_activations["value"] = output.detach()

# # Register hook on first transformer block's c_attn
# hook = model.h[0].attn.c_attn.register_forward_hook(c_attn_hook)

# # -------- Run model --------
# with torch.no_grad():
#     _ = model(input_ids)

# # Remove hook
# hook.remove()

# # Extract Q, K, V
# qkv = c_attn_activations["value"]     # [1, seq, 2304]
# hidden_size = model.config.hidden_size   # 768

# # Select the second token (index = 1)
# token_qkv = qkv[0, 1]                 # [2304]

# # Split into Q, K, V
# Q = token_qkv[:hidden_size]
# K = token_qkv[hidden_size:2*hidden_size]
# V = token_qkv[2*hidden_size:3*hidden_size]

# print("Q shape:", Q.shape)
# print("K shape:", K.shape)
# print("V shape:", V.shape)

# print("\nQ vector:\n", Q)
# print("\nK vector:\n", K)
# print("\nV vector:\n", V)

# import torch
# from transformers import GPT2Model, GPT2Tokenizer

# # Load GPT-2 small (without LM head)
# model_name = "gpt2"
# model = GPT2Model.from_pretrained(model_name)
# model.eval()

# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# # Input prompt
# input_text = "The capital of France is"
# inputs = tokenizer(input_text, return_tensors="pt")
# input_ids = inputs["input_ids"]

# # -------- Hook to capture attn_output (after attention, before c_proj) --------
# attn_output_capture = {}

# def attn_hook(module, input, output):
#     """
#     module: GPT2Attention
#     output: attn_output already combined across heads, shape [batch, seq, hidden]
#     """
#     attn_output = output[0]
#     attn_output_capture["value"] = attn_output.detach()

# # Register hook on first transformer block's attention module
# hook = model.h[0].attn.register_forward_hook(attn_hook)

# # -------- Run model --------
# with torch.no_grad():
#     _ = model(input_ids)

# hook.remove()

# # Extract attention output BEFORE c_proj
# attn_output = attn_output_capture["value"]    # [1, seq_len, hidden_size]

# # Select second token (index = 1)
# second_token_attn = attn_output[0, 1]         # [hidden_size]

# print("Attention output shape:", attn_output.shape)
# print("\nSecond-token attention output (before c_proj):\n", second_token_attn)

# import torch
# from transformers import GPT2Model, GPT2Tokenizer

# # Load GPT-2 small (without LM head)
# model_name = "gpt2"
# model = GPT2Model.from_pretrained(model_name)
# model.eval()

# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# # Input prompt
# input_text = "The capital of France is"
# inputs = tokenizer(input_text, return_tensors="pt")
# input_ids = inputs["input_ids"]

# # -------- Hook to capture output AFTER the full transformer block (layer 0) --------
# block_output_capture = {}

# def block_hook(module, input, output):
#     """
#     module: GPT2Block
#     output: hidden states after full block (attn + MLP + residuals)
#     """
#     block_output_capture["value"] = output[0].detach()

# # Register hook on whole transformer block 11
# hook = model.h[11].register_forward_hook(block_hook)

# # -------- Run model --------
# with torch.no_grad():
#     _ = model(input_ids)

# hook.remove()

# # Extract block output [1, seq_len, hidden_size]
# block_output = block_output_capture["value"]

# # Select second token (index = 1)
# second_token_output = block_output[0, 1]

# print("Layer-0 block output shape:", block_output.shape)
# print("\nSecond-token output (after full layer-0 block):\n", second_token_output)

# import torch
# from transformers import GPT2Model, GPT2Tokenizer
# import time

# # Load GPT-2 small (without LM head)
# model_name = "gpt2"
# model = GPT2Model.from_pretrained(model_name)
# model.eval()

# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# # Input prompt
# input_text = "The capital of France is"
# inputs = tokenizer(input_text, return_tensors="pt")
# input_ids = inputs["input_ids"]

# # # -------- Run model and get final hidden states --------
# with torch.no_grad():
#     outputs = model(input_ids)
#     final_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]

# # Select second token (index = 1)
# second_token_final = final_hidden[0, 1]

# print("Final output hidden state shape:", final_hidden.shape)
# print("\nFinal features for second token:\n", second_token_final)

# ======== CPU vs GPU Benchmarking ========
# print("\n" + "="*60)
# print("Benchmarking CPU vs GPU forward pass")
# print("="*60)

# # Number of warmup and benchmark runs
# num_warmup = 5
# num_runs = 20

# # -------- CPU Benchmark --------
# print("\n[CPU Benchmark]")
# model_cpu = GPT2Model.from_pretrained(model_name)
# model_cpu.eval()
# input_ids_cpu = input_ids.cpu()

# # Warmup
# for _ in range(num_warmup):
#     with torch.no_grad():
#         _ = model_cpu(input_ids_cpu)

# # Benchmark
# cpu_times = []
# for _ in range(num_runs):
#     start_time = time.perf_counter()
#     with torch.no_grad():
#         _ = model_cpu(input_ids_cpu)
#     end_time = time.perf_counter()
#     cpu_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

# cpu_mean = sum(cpu_times) / len(cpu_times)
# print(f"CPU average time: {cpu_mean:.3f} ms (over {num_runs} runs)")

# # -------- GPU Benchmark --------
# if torch.cuda.is_available():
#     print("\n[GPU Benchmark]")
#     model_gpu = GPT2Model.from_pretrained(model_name)
#     model_gpu.eval()
#     model_gpu = model_gpu.cuda()
#     input_ids_gpu = input_ids.cuda()
    
#     # Warmup
#     for _ in range(num_warmup):
#         with torch.no_grad():
#             _ = model_gpu(input_ids_gpu)
#     torch.cuda.synchronize()
    
#     # Benchmark
#     gpu_times = []
#     for _ in range(num_runs):
#         torch.cuda.synchronize()
#         start_time = time.perf_counter()
#         with torch.no_grad():
#             _ = model_gpu(input_ids_gpu)
#         torch.cuda.synchronize()
#         end_time = time.perf_counter()
#         gpu_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
#     gpu_mean = sum(gpu_times) / len(gpu_times)
#     print(f"GPU average time: {gpu_mean:.3f} ms (over {num_runs} runs)")
    
#     # Speedup
#     speedup = cpu_mean / gpu_mean
#     print(f"\n[Results]")
#     print(f"GPU is {speedup:.2f}x faster than CPU")
# else:
#     print("\nGPU not available for benchmarking")