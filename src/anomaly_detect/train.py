"""Anomaly detection training entry point (Autoencoder only).

Usage
-----
./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month.csv
./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month.csv data.train_on_negatives=true
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger

from anomaly_detect.data.anomaly_datamodule import AnomalyDataModule
from anomaly_detect.models.anomaly_module import AnomalyLitModule
from anomaly_detect.utils.instantiators import instantiate_callbacks
from anomaly_detect.utils.logging_utils import log_hyperparameters
from anomaly_detect.utils.metrics import compute_lift_at_percentiles, print_lift_table

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
    # input_dim is ??? in the config; patch it from the actual feature count
    with open_dict(cfg):
        cfg.model.net.input_dim = datamodule.n_features

    net = instantiate(cfg.model.net)
    lit = AnomalyLitModule(net=net, optimizer_cfg=cfg.model.optimizer)

    # ── Logger & callbacks ────────────────────────────────────────────────
    wandb_logger: Optional[WandbLogger] = None
    if "logger" in cfg:
        try:
            wandb_logger = instantiate(
                cfg.logger,
                name=cfg.task_name,
                tags=list(cfg.get("tags", [])),
            )
        except Exception as e:
            log.warning(f"Could not init WANDB logger: {e}")

    callbacks = instantiate_callbacks(cfg.get("callbacks", {}))

    # EarlyStopping needs val data — disable if no val split
    if datamodule.val_dataloader() is None:
        callbacks = [cb for cb in callbacks if not hasattr(cb, "monitor")]
        log.info("No val split — EarlyStopping disabled.")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer: L.Trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger if wandb_logger else False,
        callbacks=callbacks or None,
        default_root_dir=cfg.paths.output_dir,
    )

    if wandb_logger:
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

    if wandb_logger:
        wandb_logger.experiment.log(metrics)
        wandb_logger.experiment.log({
            "top_500_providers": wandb.Table(dataframe=results_df.head(500))
        })

    # ── Save outputs ──────────────────────────────────────────────────────
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "scored_providers.csv", index=False)
    log.info(f"Scored providers → {out_dir / 'scored_providers.csv'}")

    if datamodule.auroc_df is not None:
        datamodule.auroc_df.to_csv(out_dir / "feature_auroc.csv", index=False)
        log.info(f"Feature AUROC → {out_dir / 'feature_auroc.csv'}")

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
