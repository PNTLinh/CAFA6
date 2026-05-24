"""Copy CAFA6 logs/checkpoints to /kaggle/working for notebook Output.

Checks whether a branch log looks finished (saved checkpoint/final, or best_fscore
after the last training run). MF can be skipped for retrain if the log is OK.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path


def _log_candidates(data_dir: Path, branch: str) -> list[Path]:
    names = [f"{branch}.log"]
    roots = [
        data_dir / "log",
        Path("/kaggle/working/log"),
        Path("/kaggle/working/CAFA6/log"),
    ]
    out: list[Path] = []
    for root in roots:
        for name in names:
            p = root / name
            if p.is_file():
                out.append(p)
    return out


def _model_dirs(data_dir: Path) -> list[Path]:
    return [
        data_dir / "save_models",
        Path("/kaggle/working/save_models"),
        Path("/kaggle/working/CAFA6/save_models"),
    ]


def analyze_log(text: str) -> dict:
    """Heuristic: last training block in log (after restart)."""
    blocks = text.split("########start training###########")
    block = blocks[-1] if blocks else text
    has_traceback = "Traceback" in block
    saved = bool(
        re.search(r"saved (final|checkpoint):", block, re.I)
        or re.search(r"\bSAVED\b", block)
    )
    best = re.findall(r"best_fscore:\s*([\d.]+)", block)
    epochs = [int(m.group(1)) for m in re.finditer(r"epoch:\s*(\d+)", block)]
    validating = "validating" in block
    valid_done = "valid forward done" in block or "########valid metric###########" in block
    last_epoch = max(epochs) if epochs else -1

    if saved:
        status = "ok"
        note = "Đã lưu model (saved checkpoint/final)."
    elif has_traceback:
        status = "error"
        note = "Log có Traceback — train lỗi, chỉ nên lưu log tham khảo."
    elif best and (valid_done or float(best[-1]) > 0):
        status = "ok"
        note = f"Có best_fscore={best[-1]} — coi như train xong (có thể thiếu .pkl cũ)."
    elif validating and not valid_done:
        status = "incomplete"
        note = "Dừng sau 'validating' — validation chưa xong (lỗi cacul_aupr cũ?)."
    elif last_epoch >= 0:
        status = "partial"
        note = f"Chỉ thấy train đến epoch {last_epoch}, chưa rõ validation."
    else:
        status = "empty"
        note = "Không thấy block train."

    return {
        "status": status,
        "note": note,
        "saved": saved,
        "best_fscore": best[-1] if best else None,
        "last_epoch": last_epoch,
    }


def copy_branch(
    branch: str,
    data_dir: Path,
    out_log: Path,
    out_models: Path,
) -> None:
    logs = _log_candidates(data_dir, branch)
    if not logs:
        print(f"[{branch}] Không tìm thấy {branch}.log", file=sys.stderr)
        return

    src_log = max(logs, key=lambda p: p.stat().st_mtime)
    text = src_log.read_text(encoding="utf-8", errors="replace")
    info = analyze_log(text)
    print(f"[{branch}] log: {src_log} ({src_log.stat().st_size} B)")
    print(f"         status={info['status']}: {info['note']}")

    out_log.mkdir(parents=True, exist_ok=True)
    dst_log = out_log / f"{branch}.log"
    shutil.copy2(src_log, dst_log)
    print(f"         -> {dst_log}")

    copied = 0
    out_models.mkdir(parents=True, exist_ok=True)
    for models_dir in _model_dirs(data_dir):
        if not models_dir.is_dir():
            continue
        patterns = (f"*_{branch}_*.pkl", f"*{branch}*.pkl")
        seen: set[Path] = set()
        for pat in patterns:
            for f in sorted(models_dir.glob(pat)):
                if f in seen or branch not in f.name:
                    continue
                seen.add(f)
                dst = out_models / f.name
                if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(f, dst)
                print(f"         model: {f} -> {dst}")
                copied += 1
    if copied == 0:
        print(f"         (không có .pkl cho nhánh {branch})")


def _test_log_candidates(data_dir: Path, branch: str) -> list[Path]:
    name = f"test_{branch}.log"
    roots = [
        data_dir / "log",
        Path("/kaggle/working/log"),
        Path("/kaggle/working/CAFA6/log"),
    ]
    out: list[Path] = []
    for root in roots:
        p = root / name
        if p.is_file():
            out.append(p)
    return out


def copy_test_result(
    branch: str, data_dir: Path, out_test: Path, out_log: Path | None = None
) -> None:
    src_dir = data_dir / "test_result"
    if not src_dir.is_dir():
        src_dir = Path("/kaggle/working/CAFA6/test_result")
    if src_dir.is_dir():
        out_test.mkdir(parents=True, exist_ok=True)
        for pattern in (
            f"{branch}_result.json",
            f"{branch}*_pred_actual.pkl",
            f"{branch}_roc_curve.png",
        ):
            for f in src_dir.glob(pattern):
                dst = out_test / f.name
                shutil.copy2(f, dst)
                print(f"         test_result: {f.name} -> {dst}")

    logs = _test_log_candidates(data_dir, branch)
    if not logs:
        return
    src_log = max(logs, key=lambda p: p.stat().st_mtime)
    out_test.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_log, out_test / src_log.name)
    print(f"         test log: {src_log} -> {out_test / src_log.name}")
    if out_log is not None:
        out_log.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_log, out_log / src_log.name)
        print(f"         test log: {src_log} -> {out_log / src_log.name}")


def make_zip(
    out_log: Path,
    out_models: Path,
    out_test: Path,
    zip_path: Path,
    data_dir: Path | None = None,
) -> None:
    """Pack log/, save_models/, test_result/ into one zip for Kaggle download."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder, arc_prefix in (
            (out_log, "log"),
            (out_models, "save_models"),
            (out_test, "test_result"),
        ):
            if not folder.is_dir():
                continue
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    zf.write(f, arcname=f"{arc_prefix}/{f.relative_to(folder).as_posix()}")
        if data_dir is not None:
            manifest = data_dir / "log" / "_kaggle_manifest.txt"
            if manifest.is_file():
                zf.write(manifest, arcname="log/_kaggle_manifest.txt")
    print(f"[zip] wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lưu log/model CAFA6 ra Kaggle Output")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Thư mục CAFA6 (mặc định: DATA_DIR hoặc /kaggle/working/CAFA6)",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["mf", "cc", "bp"],
        choices=["mf", "cc", "bp"],
    )
    parser.add_argument(
        "--skip-train-if-ok",
        action="store_true",
        help="In danh sách nhánh status=ok (chỉ copy, không gợi ý train lại)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Tạo /kaggle/working/cafa6_output.zip sau khi copy",
    )
    parser.add_argument(
        "--zip-name",
        default="cafa6_output.zip",
        help="Tên file zip (trong /kaggle/working/)",
    )
    args = parser.parse_args()

    import os

    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "/kaggle/working/CAFA6"))
    out_log = Path("/kaggle/working/log")
    out_models = Path("/kaggle/working/save_models")
    out_test = Path("/kaggle/working/test_result")

    print("DATA_DIR:", data_dir)
    ok_skip: list[str] = []
    for branch in args.branches:
        logs = _log_candidates(data_dir, branch)
        if logs:
            info = analyze_log(logs[0].read_text(encoding="utf-8", errors="replace"))
            if info["status"] == "ok":
                ok_skip.append(branch)
        copy_branch(branch, data_dir, out_log, out_models)
        copy_test_result(branch, data_dir, out_test, out_log)

    if args.zip:
        zip_path = Path("/kaggle/working") / args.zip_name
        make_zip(out_log, out_models, out_test, zip_path, data_dir)

    print("\n=== Output ===")
    if out_log.is_dir():
        for f in sorted(out_log.glob("*.log")):
            print(f"  log/{f.name}  ({f.stat().st_size} B)")
    if out_models.is_dir():
        for f in sorted(out_models.glob("*.pkl")):
            print(f"  save_models/{f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  save_models/ (trống)")

    if args.skip_train_if_ok and ok_skip:
        print("\nCó thể bỏ qua train lại:", ", ".join(ok_skip))


if __name__ == "__main__":
    main()
