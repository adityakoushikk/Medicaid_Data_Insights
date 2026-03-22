"""Anomaly detection training entry point (Autoencoder only).

Usage
-----
./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month.csv
./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month.csv logger=csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

import lightning.pytorch as L

from anomaly_detect.data.anomaly_datamodule import AnomalyDataModule
from anomaly_detect.models.anomaly_module import AnomalyLitModule
from anomaly_detect.utils.instantiators import instantiate_callbacks
from anomaly_detect.utils.logging_utils import log_hyperparameters
from anomaly_detect.utils.metrics import compute_lift_at_percentiles, print_lift_table, build_lift_table

log = logging.getLogger(__name__)


def train(cfg: DictConfig) -> dict:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # ── DataModule ────────────────────────────────────────────────────────
    log.info("Instantiating datamodule...")
    datamodule: AnomalyDataModule = instantiate(cfg.data)
    datamodule.setup()
    log.info(f"Features: {datamodule.n_features}")

    # ── Model ─────────────────────────────────────────────────────────────
    with open_dict(cfg):
        cfg.model.net.input_dim = datamodule.n_features

    net = instantiate(cfg.model.net)
    lit = AnomalyLitModule(net=net, optimizer_cfg=cfg.model.optimizer)

    # ── Logger ────────────────────────────────────────────────────────────
    logger = None
    if cfg.get("logger"):
        try:
            logger = instantiate(cfg.logger)
        except Exception as e:
            log.warning(f"Could not init logger: {e}")

    # Check if this is a WandbLogger for wandb-specific extras
    try:
        from lightning.pytorch.loggers import WandbLogger
        is_wandb = isinstance(logger, WandbLogger)
    except ImportError:
        is_wandb = False

    callbacks = instantiate_callbacks(cfg.get("callbacks", {}))

    # EarlyStopping needs val data — disable if no val split
    if datamodule.val_dataloader() is None:
        callbacks = [cb for cb in callbacks if not hasattr(cb, "monitor")]
        log.info("No val split — EarlyStopping disabled.")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer: L.Trainer = instantiate(
        cfg.trainer,
        logger=logger if logger else False,
        callbacks=callbacks or None,
        default_root_dir=cfg.paths.output_dir,
    )

    if is_wandb:
        log_hyperparameters({"cfg": cfg, "model": lit, "trainer": trainer})

    trainer.fit(lit, datamodule)

    # ── Score all providers ───────────────────────────────────────────────
    log.info("Scoring all providers...")
    scores = lit.compute_anomaly_scores(datamodule.X_all_np)

    percentiles = list(cfg.lift_percentiles)
    metrics, results_df = compute_lift_at_percentiles(
        scores, datamodule.y_all_np, datamodule.npis_all, percentiles
    )
    metrics["n_features"] = datamodule.n_features

    if is_wandb:
        import wandb
        logger.experiment.log(metrics)
        logger.experiment.log({
            "top_500_providers": wandb.Table(dataframe=results_df.head(500))
        })
        wandb.finish()

    # ── Save outputs ──────────────────────────────────────────────────────
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "scored_providers.csv", index=False)
    log.info(f"Scored providers → {out_dir / 'scored_providers.csv'}")

    if datamodule.auroc_df is not None:
        datamodule.auroc_df.to_csv(out_dir / "feature_auroc.csv", index=False)
        log.info(f"Feature AUROC → {out_dir / 'feature_auroc.csv'}")

    lift_df = build_lift_table(metrics, percentiles)
    lift_df.to_csv(out_dir / "lift_table.csv", index=False)
    log.info(f"Lift table → {out_dir / 'lift_table.csv'}")

    print_lift_table(metrics, percentiles)
    return metrics


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    metrics = train(cfg)
    opt_metric = cfg.get("optimized_metric")
    if opt_metric and opt_metric in metrics:
        return float(metrics[opt_metric])
    return None


if __name__ == "__main__":
    main()
