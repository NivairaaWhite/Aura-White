# Example: the four-view character sheet -> a textured GLB (no GPU, no network)

`input/` is the turnaround sheet (front, back, left, right). `output/aura_character.glb` is what `aura-white sheet` made from it:

    aura-white sheet --front input/front.png --back input/back.png --left input/left.jpg --right input/right.jpg \
        --height 1.68 --faces 60000 --name aura_character --out output

**What it is:** a closed, textured, 60k-face mesh, 1.68 m tall, with a real back painted from the back picture and the sides
painted from the side pictures. Outline agreement with your four pictures: front 0.95, back 0.93, left 0.99, right 0.98.

**What it is NOT:** a finished character. Four outlines cannot separate loose white sleeves, a robe and knee-length hair, so
they merge into one smooth bell-shaped mass (see `output/aura_character_turntable.png`). Use it as a blockout / proportion
reference, or as the base for sculpting.

**To get a finished model** use a generator for the shape and let this tool paint it:

    aura-white make front=input/front.png back=input/back.png left=input/left.jpg right=input/right.jpg   # v7 back-end (Hunyuan3D-2mv, needs a GPU)
    aura-white sheet --mesh <that mesh>.glb --front input/front.png --back input/back.png --left input/left.jpg --right input/right.jpg --height 1.68

That second route was not run here (no GPU or weights in the build sandbox).
