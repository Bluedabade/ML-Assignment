"""ตรวจ prepared dataset: counts, class, corruption, unsupported และ duplicate."""
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import sys
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from PIL import Image
from src.config import CLASS_NAMES,IMAGE_EXTENSIONS,TRAIN_DIR,VAL_DIR,TEST_DIR
def inspect_dataset(quiet=False):
 counts={}; unsupported=[]; corrupted=[]; hashes=defaultdict(list); missing=[]
 for label,folder in [("Training",TRAIN_DIR),("Validation",VAL_DIR),("Testing",TEST_DIR)]:
  counts[label]={}
  for cls in CLASS_NAMES:
   d=folder/cls; files=list(d.iterdir()) if d.is_dir() else []; images=[p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]; counts[label][cls]=len(images)
   if not images: missing.append(f"{label}/{cls}")
   unsupported += [str(p.relative_to(ROOT)) for p in files if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS]
   for p in images:
    try:
     with Image.open(p) as im: im.verify()
     hashes[sha256(p.read_bytes()).hexdigest()].append(str(p.relative_to(ROOT)))
    except (OSError,ValueError): corrupted.append(str(p.relative_to(ROOT)))
 duplicates=sum(len(v)-1 for v in hashes.values() if len(v)>1); flat=[n for x in counts.values() for n in x.values()]; imbalance=(max(flat)/min(flat)) if flat and min(flat) else float("inf")
 result={"valid":not missing and not corrupted and duplicates==0,"counts":counts,"total":sum(flat),"missing":missing,"unsupported":unsupported,"corrupted":corrupted,"duplicates":duplicates,"imbalance_ratio":imbalance}
 if not quiet:
  print("DATASET CHECK\n"+"="*60)
  for split,c in counts.items(): print(f"{split:10} {sum(c.values()):5} | "+", ".join(f"{k}={v}" for k,v in c.items()))
  print(f"Total Images: {result['total']}\nMissing Classes: {missing or 'ไม่มี'}\nUnsupported Files: {len(unsupported)}\nCorrupted Images: {len(corrupted)}\nDuplicate count: {duplicates}\nClass Imbalance (max/min): {imbalance:.2f}x")
 return result
if __name__=="__main__": raise SystemExit(0 if inspect_dataset()["valid"] else 1)
