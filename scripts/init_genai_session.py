import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.msi_genai_ocr import MSIGenAIOCR


def main() -> int:
    t0 = time.perf_counter()
    out_path = os.getenv("MSI_SESSION_ID_FILE", "")
    if len(sys.argv) > 1 and sys.argv[1].strip():
        out_path = sys.argv[1].strip()

    if not out_path:
        out_path = str(Path(__file__).resolve().parents[1] / ".msi_genai_session")

    ocr = MSIGenAIOCR()
    session_id = ocr.init_session()

    try:
        elapsed = time.perf_counter() - float(t0)
        print(f"[TIMING] Initialization: {elapsed:.3f}s", flush=True)
    except Exception:
        pass

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(session_id, encoding="utf-8")

    print(f"SessionId: {session_id}", flush=True)
    print(f"MSI_SESSION_ID={session_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
