{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShell =
          with pkgs; mkShell rec {
            buildInputs = [
              python3
              python311Packages.numpy
              python311Packages.opencv4
              python311Packages.matplotlib
              python311Packages.tensorflow
              python311Packages.keras
            ];
          };
      });
}
