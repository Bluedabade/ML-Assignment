from pathlib import Path
import subprocess,sys
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8",errors="replace")
ROOT=Path(__file__).resolve().parents[1]; found=[]
for p in ROOT.rglob("*"):
 if p.is_file() and ".git" not in p.parts and p.stat().st_size>=95*1024**2: found.append((p,p.stat().st_size))
blocked=[]
for p,n in found:
 ignored=subprocess.run(["git","check-ignore","-q",str(p)],cwd=ROOT).returncode==0
 status="IGNORED" if ignored else ("BLOCK" if n>100*1024**2 else "WARN")
 print(status,f"{n/1024**2:.1f} MB",p.relative_to(ROOT))
 if n>100*1024**2 and not ignored: blocked.append(p)
print(f"พบไฟล์ตั้งแต่ 95 MB: {len(found)}; ไฟล์ใหญ่ที่ไม่ถูก ignore: {len(blocked)}"); raise SystemExit(1 if blocked else 0)
