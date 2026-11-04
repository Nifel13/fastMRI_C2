# usage: python check_h5.py C:\Users\eduba\Documents\Project\MRI\fastMRI_C2\data\brain
import sys, pathlib, shutil, h5py

root = pathlib.Path(sys.argv[1])
bad = root / "bad"; bad.mkdir(exist_ok=True)
for f in sorted(root.rglob("*.h5")):
    if bad in f.parents:
        continue
    try:
        with h5py.File(f, "r") as h:
            k = h["kspace"]
            _ = k[k.shape[0] - 1]                  # last slice of k-space
            if f.parent.name != "heldout":
                _ = h["reconstruction_rss"][-1]    # ground truth needed for train/val
        print("OK ", f.name, f"{f.stat().st_size / 1e6:.0f} MB")
    except Exception as e:
        print("BAD", f.name, f.stat().st_size, "bytes -", type(e).__name__)
        shutil.move(str(f), bad / f.name)