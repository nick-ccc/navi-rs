{
  description = "A flake for drone server development";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: let 
    pkgs = nixpkgs.legacyPackages."x86_64-linux";
  in {
    devShells."x86_64-linux".default = pkgs.mkShell {
      buildInputs = with pkgs; [
        # main rust dependencies
        cargo 
        rustc 
        rustfmt 
        clippy 
        rust-analyzer

        # cargo tools
        cargo-watch
        cargo-edit
        cargo-tarpaulin
        cargo-audit

        glib
        pkg-config
      ];
      RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
    };
  };
}
