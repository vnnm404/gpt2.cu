/* GPT-2 inference executable - C implementation */

#include <stdio.h>
#include "gpt2/gpt2.h"

config_t config = {
    .vocab_size = 50257,
    .padded_vocab_size = 50304,
    .n_layer = 12,
    .n_head = 12,
    .n_embd = 768,
    .n_positions = 1024,
    .n_ctx = 1024
};
gpt2_t model;

int main() {
    printf("GPT-2 inference\n");
    if (gpt2_initialize(&model, &config) != 0) {
        fprintf(stderr, "Failed to initialize GPT-2 model\n");
        return -1;
    }
    printf("Model initialized successfully\n");
    gpt2_free(&model);
    return 0;
}
