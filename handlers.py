from typing import Any, Optional
from dataclasses import dataclass
import copy

from torch.utils.data.dataloader import DataLoader
from ignite.engine.engine import Engine
from ignite.engine.events import Events

import pandas as pd
import wandb


class EvaluationAccumulator:
    def __init__(self):
        self.df = {
            "timestamp": [],
            "dataset": [],
            "epoch": [],
            "iteration": [],
        }

    def append_metrics(self,
                       engine: Engine,
                       metrics: dict[str, Any],
                       name: str):
        self.df["timestamp"].append(pd.Timestamp.now())
        self.df["dataset"].append(name)
        self.df["epoch"].append(engine.state.epoch)
        self.df["iteration"].append(engine.state.iteration)

        for key, value in metrics.items():
            if key in self.df:
                self.df[key].append(value)
            else:
                self.df[key] = [value]

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.df)


@dataclass
class EvaluationRunner:
    evaluator: Engine
    dataloader: DataLoader
    name: str
    logger: EvaluationAccumulator
    trainer_engine: Engine | None = None
    print_log: bool = True
    run_evaluator: bool = True

    def __call__(self, engine: Engine):
        if self.run_evaluator:
            self.evaluator.run(self.dataloader)

        d: dict[str, Any] = copy.copy(self.evaluator.state.metrics)
        self.logger.append_metrics(engine, d, self.name)

        wandb_metrics = {f"{self.name}/{k}": v for k, v in d.items()}

        if self.trainer_engine:
            trainer_metrics = self.trainer_engine.state.metrics
            for k, v in trainer_metrics.items():
                wandb_metrics[f"{self.name}/train_{k}"] = v

        if wandb.run:
            wandb.log(wandb_metrics, step=engine.state.epoch)

        if self.print_log:
            print(self.logger.get_dataframe().iloc[-1].to_dict(), flush=True)


@dataclass
class DataStreamLogger:
    dataset_name: str
    _label_data: Optional[list] = None
    _global_sample_idx: int = 0

    def attach(self, engine: Engine):
        self._label_data = []
        self._global_sample_idx = 0
        
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._log_iteration)
        engine.add_event_handler(Events.COMPLETED, self._log_final_plot)

    def _log_iteration(self, engine: Engine):
        if engine.state.output is None or not isinstance(engine.state.output, tuple):
            return
        if self._label_data is None:
            return
        
        _, y = engine.state.output
        current_batch_size = y.shape[0]
        
        for j in range(current_batch_size):
            self._label_data.append(
                [self._global_sample_idx, y[j].item()]
            )
            self._global_sample_idx += 1


    def _log_final_plot(self, engine: Engine):
        if not self._label_data or not wandb.run:
            return

        label_table = wandb.Table(
            data=self._label_data, 
            columns=["sample_index", "label_value"]
        )

        wandb.log({
            f"data_stream/{self.dataset_name}/label_plot": wandb.plot.line(
                label_table, 
                "sample_index", 
                "label_value", 
                title=f"{self.dataset_name} Label Stream"
            )
        })