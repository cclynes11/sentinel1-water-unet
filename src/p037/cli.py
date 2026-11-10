"""Command line: ``p037 <stage> --config <file.yaml> [--set key=value ...]``.

    p037 data      build the patch cache and prepare the test scenes
    p037 train     train the U-Net (resumes from last.pt if present)
    p037 evaluate  predict and score the test scenes per tile (or cached patches)
    p037 proxies   check that every training patch and test tile has proxy values
    p037 analyze   join results and proxies; tables, figures and report.md
    p037 all       evaluate -> proxies -> analyze (after training)
"""

from __future__ import annotations

import argparse
import logging

from .config import load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="experiment YAML (analysis/p037/configs/)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="p037", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("data", help="patch cache and test scenes")
    _add_common(p)
    p.add_argument("--what", choices=["cache", "scenes", "all"], default="all")
    p.add_argument("--source", default=None,
                   help="extracted dataset folder or .tar.gz archive (cache); folder with the scene GeoTIFFs (scenes). "
                        "Default: download from Hugging Face")
    p.add_argument("--scene-source", default=None, help="folder with the test scene GeoTIFFs, if different from --source")
    p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("train", help="train the model")
    _add_common(p)
    p.add_argument("--restart", action="store_true", help="ignore last.pt and start from epoch 1")

    p = sub.add_parser("evaluate", help="per-tile scores and confidence")
    _add_common(p)
    p.add_argument("--patches", choices=["train", "val"], default=None,
                   help="score cached patches of this split instead of the test scenes")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("proxies", help="check proxy coverage")
    _add_common(p)

    p = sub.add_parser("analyze", help="tables, figures, report")
    _add_common(p)
    p.add_argument("--tiles", default=None, help="per-tile results CSV (default: <run folder>/tiles.csv)")
    p.add_argument("--out", default=None, help="output folder (default: <run folder>/analysis)")

    p = sub.add_parser("all", help="evaluate, proxies, analyze")
    _add_common(p)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.set)

    if args.stage == "data":
        from .datasets import get_dataset

        ds = get_dataset(cfg)
        if args.what in ("cache", "all"):
            if ds.has_cache():
                logging.info("cache exists in %s (delete it to rebuild)", ds.cache_dir)
            else:
                ds.build_cache(args.source, workers=args.workers)
        if args.what in ("scenes", "all"):
            for event in cfg.eval_events or ds.test_events:
                ds.prepare_scene(event, args.scene_source or args.source)
    elif args.stage == "train":
        from .train import train

        train(cfg, args.device, restart=args.restart)
    elif args.stage == "evaluate":
        from .evaluate import evaluate_patches, evaluate_scenes

        if args.patches:
            evaluate_patches(cfg, args.patches, args.device)
        else:
            evaluate_scenes(cfg, args.device, overwrite=args.overwrite)
    elif args.stage == "proxies":
        from .proxies import check_coverage

        check_coverage(cfg)
    elif args.stage == "analyze":
        from .analyze import run_analysis

        print(run_analysis(cfg, args.tiles, args.out))
    elif args.stage == "all":
        from .analyze import run_analysis
        from .evaluate import evaluate_scenes
        from .proxies import check_coverage

        evaluate_scenes(cfg, args.device)
        check_coverage(cfg)
        print(run_analysis(cfg))


if __name__ == "__main__":
    main()
