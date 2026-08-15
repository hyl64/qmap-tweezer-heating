#!/usr/bin/env python
"""Run the full qmap verification suite.  Exit 0 iff everything passes.

Usage:
    ./.venv/bin/python run_verify.py [--quick] [--only t1,t2,...]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmap.verify import run_all

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grids/steps")
    ap.add_argument("--only", default=None, help="comma-separated test names")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    ok = run_all(quick=args.quick, only=only)
    sys.exit(0 if ok else 1)
