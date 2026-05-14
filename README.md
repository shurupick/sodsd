# SODsd

TODO:
- [ ] Дописать upscaling background
- [ ] Вращение bacground по метаданным в названии объекта. Например, 

For generate pfm image from pfm stl do:

```bash
python sodsd/render_stl_euler_sweep.py --stl /Users/akrylov/Desktop/Аспирантура/pfm/models3D/pfm-1v3.stl --rz 0:360:15 --camera top
```

Natural green color variants for the mine:

```bash
# fixed preset
python sodsd/render_stl_euler_sweep.py --stl /path/to/pfm.stl --color olive_green

# one random natural green shade for the whole run
python sodsd/render_stl_euler_sweep.py --stl /path/to/pfm.stl --randomize-color once --color-seed 42

# a different natural green shade for every rendered angle
python sodsd/render_stl_euler_sweep.py --stl /path/to/pfm.stl --randomize-color per-frame --color-seed 42
```

Generate raw backgrounds:

```bash
python sodsd/generate_data.py --count 20 --outdir data/raw --seed 42
```

Rotate generated backgrounds using `rx/ry/rz` from mine render filenames:

```bash
python sodsd/3dBackgroundRotate.py \
  --backgrounds-dir data/raw \
  --mines-dir data/pfm \
  --outdir data/interim/backgrounds_rotated \
  --order ZXY
```

The rotated background is saved with the same filename as the mine render, for example
`view_rx-40.0_ry0.0_rz40.0.png`.
