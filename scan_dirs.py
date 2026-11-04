# usage: python scan_dirs.py "C:\Users\eduba\Documents\Project\MRI\fastMRI_C2\data\brain"
import sys, pathlib, h5py

root = pathlib.Path(sys.argv[1])
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    entries = sorted(d.iterdir())
    print(f"\n{d.name}: {len(entries)} entries")
    for f in entries:
        if f.is_dir():
            print("  DIR  ", f.name); continue
        try:
            with h5py.File(f, "r") as h:
                n = h["kspace"].shape[0]
            print(f"  ok    {f.name}  {f.stat().st_size/1e6:.0f} MB  {n} slices")
        except Exception as e:
            print(f"  BAD   {f.name}  {f.stat().st_size} bytes  {type(e).__name__}")