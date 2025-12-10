{ pkgs, ... }:

let
  baseUrl = "https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main";
  weightsUrl = "https://huggingface.co/datasets/mchrcmu/gpt2cu-data/resolve/main";
in
pkgs.stdenvNoCC.mkDerivation {
  pname = "gpt2cu-data";
  version = "1";
  src = null;
  phases = [ "installPhase" ];

  installPhase = ''
    mkdir -p $out/models
    mkdir -p $out/data/tinyshakespeare
    mkdir -p $out/data/hellaswag

    # Model
    cp ${pkgs.fetchurl {
      url = "${baseUrl}/gpt2_124M_debug_state.bin";
      sha256 = "sha256-qAxESO7UfFYTFKIYnmLEmyx0d8Jl/C8VyEcNGEAWk9k=";
    }} $out/models/gpt2_124M_debug_state.bin

    cp ${pkgs.fetchurl {
        url = "${weightsUrl}/gpt2-124M-weights.bin";
        sha256 = "sha256-0AnLBklY4DnAQeqsJ2dQ/FucrIjrwDzjKDdva+5AhTM=";
    }} $out/models/gpt2-124M-weights.bin

    # Tokenizer
    cp ${pkgs.fetchurl {
      url = "${baseUrl}/gpt2_tokenizer.bin";
      sha256 = "sha256-bzq8IeRE5OgwDiJfTgPaSOoSHPF+MPZwCbja16ZsLxM=";
    }} $out/models/gpt2_tokenizer.bin

    # Val sets
    cp ${pkgs.fetchurl {
      url = "${baseUrl}/tiny_shakespeare_val.bin";
      sha256 = "sha256-/pnbcg3HyD5pSAbU4EepUpCUEdodrM3kzMLlX0CIKmI=";
    }} $out/data/tinyshakespeare/tiny_shakespeare_val.bin

    cp ${pkgs.fetchurl {
      url = "${baseUrl}/hellaswag_val.bin";
      sha256 = "sha256-WVCbpAHoIPLNi1eRQTVOjrwUog+OLMGB8YNvHQ5Z65k=";
    }} $out/data/hellaswag/hellaswag_val.bin
  '';
}
