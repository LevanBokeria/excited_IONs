{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  name = "genai-python-shell";

  buildInputs = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
    docling
  ];

  shellHook = ''
    export PYTHONPATH=$PWD:$PYTHONPATH
    # if [ ! -d .venv ]; then
    #   echo "Creating virtualenv in .venv..."
    #   python -m venv .venv
    #   source .venv/bin/activate
    #   pip install --upgrade pip
    #   pip install google-genai pydantic
    # else
    # fi
    source .venv/bin/activate

    echo "Activated Python env. Run your script with: python myscript.py"
  '';
}
