"""`gold` CLI — fetch / features / train / predict / backtest.

The pipeline runs in this order; each step reads the previous step's output:

    gold fetch  →  gold features  →  gold train  →  gold predict
                                  └→  gold backtest   (honest out-of-sample eval)
"""

from __future__ import annotations

import argparse

from . import config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="gold", description="Daily gold-price forecaster")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="download raw price history → data/raw/")
    pf.add_argument("--ticker", default=None, help=f"default {config.PRIMARY_TICKER}")
    pf.add_argument("--start", default=None, help=f"default {config.START_DATE}")
    pf.add_argument("--cross", action="store_true",
                    help=f"also fetch the cross-check ticker ({config.CROSS_TICKER})")

    pi = sub.add_parser("import-csv", help="import a local OHLC(V) CSV → data/raw/ (offline escape hatch)")
    pi.add_argument("path", help="path to a CSV with at least Date + Close columns")
    pi.add_argument("--ticker", default=None, help=f"label for the series (default {config.PRIMARY_TICKER})")

    ps = sub.add_parser("spot", help="live current spot price via gold-api.com (no key)")
    ps.add_argument("--symbol", default="XAU", help="metal symbol (default XAU gold)")

    pft = sub.add_parser("features", help="build feature matrix → data/processed/")
    pft.add_argument("--ticker", default=None, help=f"series to featurize (default {config.PRIMARY_TICKER})")
    pft.add_argument("--macro", action="store_true", help="also add exogenous features from data/raw/macro.parquet")

    ptr = sub.add_parser("train", help="fit XGBoost on pre-test window → models/")
    ptr.add_argument("--test-start", default=None, help=f"train on rows before this date (default {config.TEST_START})")

    ppr = sub.add_parser("predict", help="next-day point forecast")
    ppr.add_argument("--ticker", default=None, help=f"series to forecast (default {config.PRIMARY_TICKER})")

    pbt = sub.add_parser("backtest", help="walk-forward eval vs random-walk baseline → reports/")
    pbt.add_argument("--test-start", default=None, help=f"out-of-sample window start (default {config.TEST_START})")

    pst = sub.add_parser("strategy", help="Triple Screen strategy backtest (win-rate / expectancy) → reports/")
    pst.add_argument("--ticker", default=None, help=f"series (default {config.PRIMARY_TICKER})")
    pst.add_argument("--rr", type=float, default=1.5, help="reward:risk multiple (default 1.5)")

    args = p.parse_args(argv)

    if args.cmd == "fetch":
        from .data.fetch import fetch
        fetch(args.ticker, args.start)
        if args.cross:
            fetch(config.CROSS_TICKER, args.start)

    elif args.cmd == "import-csv":
        from .data.fetch import import_csv
        import_csv(args.path, args.ticker)

    elif args.cmd == "spot":
        from .live import spot
        s = spot(args.symbol)
        print(f"{s['name']} ({s['symbol']})  ${s['price']:,.2f}  as of {s['updated_at']}")

    elif args.cmd == "features":
        if args.macro:
            config.USE_MACRO = True
        from .features.build import build_features
        df = build_features(args.ticker)
        print(f"features: {df.shape[0]} rows × {df.shape[1] - 2} feature cols  "
              f"({df.index.min().date()} → {df.index.max().date()})  → {config.FEATURES_PATH.name}")

    elif args.cmd == "train":
        from .models.train import train
        train(test_start=args.test_start)

    elif args.cmd == "predict":
        from .models.predict import predict_next
        predict_next(args.ticker)

    elif args.cmd == "backtest":
        from .models.backtest import backtest
        backtest(test_start=args.test_start)

    elif args.cmd == "strategy":
        import json

        from .data.load import load_raw
        from .indicators import resample_weekly
        from .strategy import triple_screen_backtest
        d = load_raw(args.ticker)
        rep = triple_screen_backtest(d, resample_weekly(d), rr=args.rr)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.REPORTS_DIR / "strategy.json").write_text(json.dumps(rep, indent=2))
        print(f"Triple Screen backtest  {rep['period']}  (R:R {rep['rr']})")
        if rep["n_trades"]:
            print(f"  trades={rep['n_trades']}  win={rep['win_rate']:.0%}  "
                  f"expectancy={rep['expectancy']:+.2%}/trade  PF={rep['profit_factor']:.2f}")
            print(f"  total={rep['total_return']:+.0%}  maxDD={rep['max_drawdown']:.0%}  "
                  f"| buy&hold={rep['buy_hold']:+.0%}  vs-random edge={rep['edge_vs_random']:+.2%}")
            verdict = "NO demonstrated edge" if rep["expectancy"] <= 0 or rep["edge_vs_random"] <= 0 else "beats random + positive"
            print(f"  → {verdict}. Method-derived, not advice; one parameterization only.")
        else:
            print("  no trades generated")


if __name__ == "__main__":
    main()
