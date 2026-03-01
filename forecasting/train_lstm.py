import argparse
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime

# 注入 D 盘 CUDA 动态链接库路径 (Python 3.8+ 必须)
cuda_bin = r"D:\NVIDIA\CUDA\Development\v11.2\bin"
if os.path.exists(cuda_bin) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(cuda_bin)

from forecasting.models.lstm.trainer import TrainConfig, train_loop


# ========== 可直接修改的训练参数（改完保存后直接运行本文件即可开始训练）==========

CLEAN_PYCACHE = True

RESET = False

LOAD_EXISTING_WEIGHTS = True

AUTOREGRESSIVE_TRAINING = True

NORMALIZE_HK_SYMBOLS_FILE = True

AUTO_MATCH_A_LIMIT_TO_HK = True 

AUTO_RESET_ON_UNIVERSE_MISMATCH = True 

WIPE_ALL_LSTM_ARTIFACTS = False 

MARKETS = ('A', 'HK')

A_BOARD = 'all'
A_LIMIT = 200
A_STOCKS_FILE = ''

HK_STOCKS_FILE = str((Path(__file__).resolve().parent / 'hk_symbols.txt'))

STEPS = 1800
SAVE_EVERY = 5

LOOKBACK = 30
TOTAL_DAYS = 50

EPOCHS_PER_STOCK = 1
BATCH_SIZE = 16

UNITS = 128
LAYERS = 3
DROPOUT = 0.5
LEARNING_RATE = 0.001


def _clean_pycache_dirs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return 0
    for p in root.rglob('__pycache__'):
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except Exception:
            pass
    return removed


def _version_hint() -> None:
    v = sys.version_info
    if (v.major, v.minor) != (3, 10):
        print(f"[WARN] current interpreter: Python {v.major}.{v.minor}.{v.micro}; recommended: Python 3.10.x")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] GPU ACCELERATION ENABLED: {gpus[0].name}")
        else:
            print("[WARN] NO GPU DETECTED - FALLING BACK TO CPU")
    except Exception:
        pass


def _normalize_hk_code(code: str) -> str:
    c = (code or '').strip()
    if not c:
        return ''
    c = c.replace('.HK', '').replace('.hk', '')
    digits = ''.join([ch for ch in c if ch.isdigit()])
    if digits:
        return digits.zfill(5)
    return c


def _normalize_hk_symbols_file(path: str) -> int:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    raw = p.read_text(encoding='utf-8', errors='ignore')
    in_lines = [ln.strip() for ln in raw.splitlines()]

    out: list[str] = []
    seen = set()
    removed = 0
    for ln in in_lines:
        if not ln or ln.startswith('#'):
            continue
        code = _normalize_hk_code(ln)
        if not code:
            continue
        if code in seen:
            removed += 1
            continue
        seen.add(code)
        out.append(code)

    new_text = '\n'.join(out) + ('\n' if out else '')
    old_text = raw.replace('\r\n', '\n')
    if old_text != new_text:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = p.with_name(p.name + f'.bak_{ts}')
        backup.write_text(raw, encoding='utf-8')
        p.write_text(new_text, encoding='utf-8')
        print(
            f"[INFO] normalized HK symbols: {p.name} unique={len(out)} removed_duplicates={removed} backup={backup.name}",
            flush=True,
        )
    return int(len(out))


def _wipe_lstm_artifacts() -> None:
    root = Path(__file__).resolve().parent
    model_dir = root / 'models' / 'lstm'
    training_dir = root / 'training_data'

    removed_files = 0
    removed_dirs = 0

    for fp in [model_dir / 'state.json', model_dir / 'history.jsonl', model_dir / 'meta.json']:
        try:
            if fp.exists() and fp.is_file():
                fp.unlink()
                removed_files += 1
        except Exception:
            pass

    ckpt_dir = model_dir / 'checkpoints'
    if ckpt_dir.exists() and ckpt_dir.is_dir():
        try:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            removed_dirs += 1
        except Exception:
            pass

    print(
        f"[INFO] wiped LSTM artifacts: removed_files={removed_files} removed_dirs={removed_dirs}",
        flush=True,
    )


def main(argv=None):
    _version_hint()

    default_hk = HK_STOCKS_FILE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--save-every', type=int, default=SAVE_EVERY)
    parser.add_argument('--lookback', type=int, default=LOOKBACK)
    parser.add_argument('--total-days', type=int, default=TOTAL_DAYS)
    parser.add_argument('--epochs-per-stock', type=int, default=EPOCHS_PER_STOCK)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--units', type=int, default=UNITS)
    parser.add_argument('--layers', type=int, default=LAYERS)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--markets', default=','.join(MARKETS))
    parser.add_argument('--a-board', default=A_BOARD)
    parser.add_argument('--a-limit', type=int, default=A_LIMIT)
    parser.add_argument('--a-stocks-file', default=A_STOCKS_FILE)
    parser.add_argument('--hk-stocks-file', default=default_hk)
    parser.add_argument('--reset', action='store_true', default=bool(RESET))

    args, _ = parser.parse_known_args(argv)

    if CLEAN_PYCACHE:
        removed = _clean_pycache_dirs(Path(__file__).resolve().parent)
        if removed:
            print(f"[INFO] removed __pycache__ dirs: {removed}")

    if WIPE_ALL_LSTM_ARTIFACTS:
        _wipe_lstm_artifacts()

    markets = tuple([s.strip().upper() for s in str(args.markets).split(',') if s.strip()])

    hk_file = str(args.hk_stocks_file or '').strip()
    hk_count = 0
    if 'HK' in markets:
        if not hk_file:
            raise ValueError('HK training requires hk_symbols.txt or --hk-stocks-file')
        if not Path(hk_file).exists():
            raise FileNotFoundError(hk_file)
        hk_count = _normalize_hk_symbols_file(hk_file)

    effective_a_limit = int(args.a_limit)
    if 'A' in markets and 'HK' in markets and AUTO_MATCH_A_LIMIT_TO_HK and (not str(args.a_stocks_file or '').strip()):
        if hk_count > 0:
            effective_a_limit = int(hk_count)
            print(f"[INFO] auto match A_LIMIT to HK: A_LIMIT={effective_a_limit}", flush=True)

    cfg = TrainConfig(
        lookback=int(args.lookback),
        total_days=int(args.total_days),
        epochs_per_stock=int(args.epochs_per_stock),
        batch_size=int(args.batch_size),
        units=int(args.units),
        layers=int(args.layers),
        dropout=float(args.dropout),
        learning_rate=float(args.lr),
        save_every=int(args.save_every),
        markets=tuple([s.strip().upper() for s in str(args.markets).split(',') if s.strip()]),
        a_board=str(args.a_board),
        a_limit=int(effective_a_limit),
        a_stocks_file=str(args.a_stocks_file or ''),
        hk_stocks_file=hk_file,
        steps=int(args.steps),
        reset=(True if WIPE_ALL_LSTM_ARTIFACTS else bool(args.reset)),
        load_existing_weights=bool(LOAD_EXISTING_WEIGHTS),
        autoregressive_training=bool(AUTOREGRESSIVE_TRAINING),
    )

    try:
        train_loop(cfg)
    except ValueError as e:
        msg = str(e)
        if 'universe mismatch' in msg:
            if AUTO_RESET_ON_UNIVERSE_MISMATCH and (not bool(cfg.reset)):
                print(f"[WARN] {msg}")
                print("[INFO] auto reset enabled: rebuild progress and continue", flush=True)
                cfg.reset = True
                train_loop(cfg)
                return
            print(f"[ERROR] {msg}")
            return
        raise


if __name__ == '__main__':
    main()
