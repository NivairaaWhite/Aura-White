#!/usr/bin/env sh
# Linux / macOS installer:  ./install.sh   or   ./install.sh cu126   (NVIDIA build of PyTorch)
# Add "studio" to also install the texturing extras:  ./install.sh cu126 studio
# Add "3d" for the TRUE-3D engine (text / pictures -> a real closed asset, runs in <= 8 GB):  ./install.sh cu126 3d
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
STUDIO=0
GEN3D=0
for a in "$@"; do [ "$a" = "studio" ] && STUDIO=1; [ "$a" = "3d" ] && GEN3D=1; done
[ -n "$1" ] && [ "$1" != "studio" ] && [ "$1" != "3d" ] && pip install torch --index-url "https://download.pytorch.org/whl/$1"
pip install -e ".[recommended]" || pip install -e .
[ "$STUDIO" = "1" ] && { pip install -r requirements-studio.txt || echo "some optional extras failed - built-in fallbacks are used"; }
python -m aura_white doctor
[ "$GEN3D" = "1" ] && { pip install -r requirements-3d.txt || echo "some 3d extras failed - run aura-white doctor3d to see what is missing"; .venv/bin/aura-white setup --hunyuan || echo "could not fetch Hunyuan3D-2 (offline?). Unzip Hunyuan3D-2-main into third_party/ by hand - see third_party/README.md"; .venv/bin/aura-white doctor3d; }
echo "Next:  .venv/bin/aura-white make picture.png     |     .venv/bin/aura-white make \"a bronze dragon\"     |"
echo "Next:  .venv/bin/aura-white serve     |     .venv/bin/aura-white studio art.png"
