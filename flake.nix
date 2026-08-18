{
  description = "Reproducible development environment for strategy2048";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.python313
            pkgs.uv
            pkgs.hyperfine
            pkgs.git
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
          UV_PYTHON_DOWNLOADS = "never";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
        };
      });
      checks = forAllSystems (pkgs: {
        python-import = pkgs.runCommand "strategy2048-import-check" {
          nativeBuildInputs = [ pkgs.python313 ];
        } ''
          export PYTHONPATH=${self}/src
          python -c 'import strategy2048'
          touch $out
        '';
      });
    };
}
