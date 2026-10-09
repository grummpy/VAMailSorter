"""Command line for train, evaluate, and classify.

``classify`` reads .eml files or an mbox and writes a report. It does not
reply, delete, or move mail. ``--gmail`` is off unless you pass it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vams import __version__
from vams.data import load_emails, require_split
from vams.evaluate import evaluate
from vams.gmail_fetch import DEFAULT_CLIENT, DEFAULT_TOKEN, fetch_messages
from vams.mailio import load_path
from vams.models import load_bundle, save_bundle, train
from vams.report import classify_messages, write_csv, write_html


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vams", description="Local VA/DFAS/VGLI mail triage aid")
    parser.add_argument("--version", action="version", version=f"vams {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser(
        "train", help="Fit on the train split; choose thresholds on validation"
    )
    train_parser.add_argument("--data", default="data/synthetic/emails.csv")
    train_parser.add_argument("--out", default="artifacts/model.joblib")
    train_parser.add_argument("--seed", type=int, default=42)

    eval_parser = sub.add_parser("evaluate", help="Score the held-out test split once")
    eval_parser.add_argument("--model", default="artifacts/model.joblib")
    eval_parser.add_argument("--data", default="data/synthetic/emails.csv")
    eval_parser.add_argument("--out", default="reports")
    eval_parser.add_argument("--model-card", default="docs/model_card.md")

    classify_parser = sub.add_parser("classify", help="Classify .eml files or an mbox export")
    classify_parser.add_argument("path", nargs="?", help="File or directory of .eml/.mbox mail")
    classify_parser.add_argument(
        "--gmail", action="store_true", help="Read-only Gmail fetch (off by default)"
    )
    classify_parser.add_argument("--model", default="artifacts/model.joblib")
    classify_parser.add_argument("--csv", default="vams-report.csv")
    classify_parser.add_argument("--html", default="vams-report.html")
    classify_parser.add_argument("--client-secrets", default=str(DEFAULT_CLIENT))
    classify_parser.add_argument("--token", default=str(DEFAULT_TOKEN))
    classify_parser.add_argument("--query", default="")
    classify_parser.add_argument("--max", type=int, default=20)
    return parser


def _train(args) -> int:
    frame = load_emails(args.data)
    bundle = train(require_split(frame, "train"), require_split(frame, "val"), seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, out)
    summary = {
        "seed": bundle.seed,
        "best_C": bundle.best_C,
        "best_svm_C": bundle.best_svm_C,
        "best_nb_alpha": bundle.best_nb_alpha,
        "threshold": bundle.threshold,
        "threshold_policy": bundle.threshold_policy,
        "threshold_selected_on": bundle.threshold_selected_on,
        "validation_precision_action": bundle.threshold_val_precision,
        "validation_recall_action": bundle.threshold_val_recall,
        "validation_f2_action": bundle.threshold_val_f2,
        "suspicious_threshold": bundle.suspicious_threshold,
        "suspicious_policy": bundle.suspicious_policy,
        "cv_n_splits": bundle.cv_n_splits,
        "cv": bundle.cv_results,
        "model": str(out),
    }
    summary_path = out.parent / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Trained logistic C={bundle.best_C} threshold={bundle.threshold:.2f} "
        f"({bundle.threshold_policy} on validation). "
        f"Validation recall={bundle.threshold_val_recall:.3f} "
        f"precision={bundle.threshold_val_precision:.3f}. Wrote {out}"
    )
    return 0


def _evaluate(args) -> int:
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No model at {model_path}. Run `vams train` first.", file=sys.stderr)
        return 2
    frame = load_emails(args.data)
    bundle = load_bundle(model_path)
    metrics = evaluate(
        bundle,
        require_split(frame, "test"),
        out_dir=Path(args.out),
        model_card_path=Path(args.model_card),
        n_train=int((frame["split"] == "train").sum()),
        n_val=int((frame["split"] == "val").sum()),
    )
    held = metrics["held_out"]["tfidf_logreg"]
    print(
        "Held-out ACTION_NEEDED "
        f"precision={held['precision_action_needed']:.3f} "
        f"recall={held['recall_action_needed']:.3f} "
        f"f2={held['f2_action_needed']:.3f} "
        f"brier={held['brier']:.3f} ece={held['ece_10']:.3f}"
    )
    print(f"Wrote {args.out} and {args.model_card}")
    return 0


def _classify(args) -> int:
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No model at {model_path}. Run `vams train` first.", file=sys.stderr)
        return 2
    if args.gmail and args.path:
        print("Pass either a path or --gmail, not both.", file=sys.stderr)
        return 2
    if not args.gmail and not args.path:
        print("Provide a path to .eml/.mbox mail, or pass --gmail.", file=sys.stderr)
        return 2
    bundle = load_bundle(model_path)
    if args.gmail:
        messages = fetch_messages(
            client_secrets=Path(args.client_secrets),
            token_path=Path(args.token),
            query=args.query or None,
            max_results=args.max,
        )
    else:
        messages = load_path(Path(args.path))
    if not messages:
        print("No messages found.", file=sys.stderr)
        return 2
    rows = classify_messages(bundle, messages)
    write_csv(rows, Path(args.csv))
    write_html(rows, Path(args.html))
    print(f"Classified {len(rows)} message(s). Wrote {args.csv} and {args.html}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "train":
        return _train(args)
    if args.cmd == "evaluate":
        return _evaluate(args)
    if args.cmd == "classify":
        return _classify(args)
    parser.error(f"Unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
