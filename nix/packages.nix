{ pkgs, cudaPackages, gccHost, nixgl, system, src, weights? ../. }:
# { pkgs, cudaPackages, gccHost, nixgl, src ? ../. }:
let
  colors = {
    yellow = "\\033[33m";
    reset  = "\\033[0m";
  };
in
{
  inference = rec {
    build = pkgs.stdenvNoCC.mkDerivation {
      pname = "gpt2cu";
      version = "0.1";
      inherit src;

      nativeBuildInputs = [
        pkgs.cmake
        pkgs.gnumake
        cudaPackages.cudatoolkit
        gccHost
      ];

      configurePhase = ''
        cmake -B build -S . \
          -DCMAKE_C_COMPILER=${gccHost}/bin/gcc \
          -DCMAKE_CXX_COMPILER=${gccHost}/bin/g++ \
          -DCMAKE_CUDA_COMPILER=${cudaPackages.cudatoolkit}/bin/nvcc \
          -DCMAKE_BUILD_TYPE=Release
      '';

      buildPhase = ''
        cmake --build build -j
      '';

      installPhase = ''
        mkdir -p $out/bin
        cp build/programs/inference $out/bin/inference
      '';
    };

    run = pkgs.writeShellScriptBin "inference" ''
      if [ "''${NIX_ENFORCE_NO_NATIVE:-0}" = "1" ]; then
          printf "${colors.yellow}WARNING:${colors.reset} native CPU features disabled via NIX_ENFORCE_NO_NATIVE=1.\n"
          printf "         If CUDA_ARCHITECTURES was not set explicitly, NVCC may fall back to generic PTX.\n"
      fi
      export DATA_ROOT=${weights}
      exec ${nixgl}/bin/nixglhost ${build}/bin/inference "$@"
    '';
  };
}
