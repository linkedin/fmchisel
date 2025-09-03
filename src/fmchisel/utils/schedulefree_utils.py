import itertools
from collections.abc import Iterable
from typing import Any, Callable, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from fmchisel.optimizers.adamw_schedulefree import AdamWScheduleFree

_DEFAULT_RETRACK_STEPS = 20


class ScheduleFreeLightningModule(pl.LightningModule):
    """
    LightningModule base class with necessary hooks implemented in order to use ScheduleFree
    optimizers.

    When using ScheduleFree optimizers, the final weight saved and used for validation is an exponential moving
    average of the past iterates, thus there're some paremeter swapping required before training, before validation
    and after training loop ends. This helper function dynamically adds necessary hooks to `LightningModule` so
    Lightning trainer will fire the hooks for flipping the parameters at the right time.

    In the scenario where the model contains BatchNorm, this helper allows you to retrack BatchNorm stats which is
    required for using ScheduleFree -- once the parameters are flipped, the input activation to BN will be under a
    different distribution.

    :param use_schedulefree: whether is using ScheduleFree optimizer. Turning this off essentially means using the
    vanilla `LightningModule` without any additional hook introduced by this module.
    :param retrack_batchnorm_stats: whether to retract BatchNorm moving avg/var. This has to be True if you have
    BatchNorm in your model (no need to set it if you're using LayerNorm, RMSNorm etc. that doesn't maintain a moving
    average). False by default. NOTE: when this is True, you must implement the `retrack_fn` method.
    :param retrack_steps: the number of forward steps to run for retracking. Default to 20 if not specified.
    """

    def __init__(
        self,
        use_schedulefree: bool = True,
        retrack_batchnorm_stats: bool = False,
        retrack_steps: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.use_schedulefree = use_schedulefree
        self.retrack_batchnorm_stats = retrack_batchnorm_stats
        if retrack_steps and not retrack_batchnorm_stats:
            raise ValueError("retrack_steps is only valid when retrack_batchnorm_stats is True")
        self.retrack_steps = retrack_steps or _DEFAULT_RETRACK_STEPS

    def retrack_fn(self, batch):  # noqa
        """
        Define the forward pass of the model here. Retrack the statistics of BatchNorm layer in model. Example:
        ```
        def retrack_fn(self, batch):
            self.trainer.strategy.model(
                batch['input_ids'].to(self.device),
                batch['attention_mask'].to(self.device),
            )
        ```
        Several notes:
        1. If using FSDP or DeepSpeed, for performing forward pass, please use `self.trainer.strategy.model()`
        instead of directly `self()` or `self.model()` as the latter ones don't give you the actual FSDP/DeepSpeed
        wrapped module.
        2. Please explicitly move the input tensors to device here.
        """
        if self.retrack_batchnorm_stats:
            raise NotImplementedError("retrack_fn method must be implemented when retrack_batchnorm_stats is True")

    def _switch_to_train(self):
        self.optimizers().train()

    def _switch_to_eval(self):
        if self.retrack_batchnorm_stats:
            self.train()
            self.optimizers().eval()
            with torch.no_grad():
                for batch in itertools.islice(self.trainer.train_dataloader, self.retrack_steps):
                    self.retrack_fn(batch)
            self.eval()
        else:
            self.optimizers().eval()

    def on_validation_model_eval(self) -> None:
        if self.use_schedulefree:
            self._switch_to_eval()
        super().on_validation_model_eval()

    def on_validation_model_train(self) -> None:
        if self.use_schedulefree:
            self._switch_to_train()

    def on_test_model_eval(self) -> None:
        if self.use_schedulefree:
            self._switch_to_eval()
        super().on_test_model_eval()

    def on_test_model_train(self) -> None:
        if self.use_schedulefree:
            self._switch_to_train()

    def on_predict_model_eval(self) -> None:
        if self.use_schedulefree:
            self._switch_to_eval()
        super().on_predict_model_eval()

    def on_train_epoch_end(self) -> None:
        if self.use_schedulefree:
            self._switch_to_eval()
        super().on_train_epoch_end()

    def on_train_epoch_start(self) -> None:
        if self.use_schedulefree:
            self._switch_to_train()
        super().on_train_epoch_start()


class ScheduleFreeSwitchCallback(pl.Callback):
    """
    Callback for switching between training and evaluation mode when using ScheduleFree optimizers.

    When using ScheduleFree optimizers, the final weight being saved and used for validation is an exponential moving
    average of the past iterates used in training, thus there're some paremeter swapping required before training,
    before validation and after training loop ends. This callback adds necessary hooks so that Lightning trainer
    will fire the hooks for flipping the parameters at the right time.

    In the scenario where the model contains BatchNorm, this helper allows you to retrack BatchNorm stats which is
    required for using ScheduleFree -- once the parameters are flipped, the input activation to BN will be under a
    different distribution.

    :param retrack_batchnorm_stats: whether to retract BatchNorm moving avg/var. This has to be True if you have
    BatchNorm in your model (no need to set it if you're using LayerNorm, RMSNorm etc. that doesn't maintain a moving
    average). False by default.
    :param retrack_fn: a function that simply define the forward pass. This function should have one input which is
    the batch example returned from your train dataloader (it can be a dict, a list, or anything, depending on your
    Dataset implementation). It doesn't have to return anything. This function has to be specified if `retrack_batchnorm_stats`
    is True. Example:
        ```
        def retrack_fn(batch, module):
            module(
                batch['input_ids'].to(module.device),
                batch['attention_mask'].to(module.device),
            )
        ```
        Several notes:
        1. If using FSDP or DeepSpeed, for performing forward pass, please use `module.trainer.strategy.model()`
        instead of directly `module()` as it doesn't give you the actual FSDP/DeepSpeed wrapped module.
        2. Please explicitly move the input tensors to device here.
    :param retrack_steps: the number of forward steps to run for retracking. Default to 20 if not specified.

    """

    def __init__(
        self,
        retrack_batchnorm_stats: bool = False,
        retrack_fn: Optional[Callable] = None,
        retrack_steps: Optional[int] = None,
    ):
        super().__init__()
        self.retrack_batchnorm_stats = retrack_batchnorm_stats
        if retrack_batchnorm_stats and not retrack_fn:
            raise ValueError("retrack_fn must be provided when retrack_batchnorm_stats is True")
        self.retrack_steps = retrack_steps or _DEFAULT_RETRACK_STEPS
        # [ NOTE: Why do we use this lazy-switch way to flip param to train mode after eval ]
        # Lightning trainer has internal logic that ensures checkpoint callbacks are always placed
        # to the end of callback list, and checkpoints are saved during `on_validation_end`. Thus
        # we need to make sure that params are in eval mode when `on_validation_end` of callbacks
        # are called, and flipped to train immediately after that. However there's no dedicated callback hook
        # in lightning that will be guaranteed to get triggered right after `on_validation_end` (we can't simply
        # rely on `on_train_epoch_start` because in many cases people will set validation check interval to be
        # a number of steps, rather than a full epoch). Thus the only option that works for all case regardless
        # of whether validation is per-epoch or within-epoch is to do it prior to training batch.
        self._should_lazy_switch_train = False

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # below logics make sure checkpoint callbacks are placed after SF switch callback.
        # This is just an additional guardrail on top of https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/trainer/connectors/callback_connector.py#L198-L201
        for _, callback in enumerate(trainer.callbacks):
            if callback is self:
                return
            if isinstance(callback, ModelCheckpoint):
                raise ValueError(
                    "Please make sure to put ScheduleFreeSwitchCallback prior to "
                    "checkpoint callback in order for saving the correct checkpoint "
                    "with ScheduleFree optimizer."
                )
        self._should_lazy_switch_train = False

    def _switch_to_train(self, module: "pl.LightningModule"):
        opt = module.optimizers()
        opt_list = opt if isinstance(opt, Iterable) else [opt]
        for opt in opt_list:
            if isinstance(opt, AdamWScheduleFree):
                opt.train()

    def _switch_to_eval(self, module: "pl.LightningModule"):
        def _flip_opt_to_eval():
            opt = module.optimizers()
            opt_list = opt if isinstance(opt, Iterable) else [opt]
            for opt in opt_list:
                if isinstance(opt, AdamWScheduleFree):
                    opt.eval()

        if self.retrack_batchnorm_stats:
            module.train()
            _flip_opt_to_eval()
            with torch.no_grad():
                for batch in itertools.islice(self.trainer.train_dataloader, self.retrack_steps):
                    self.retrack_fn(batch, module)
            module.eval()
        else:
            _flip_opt_to_eval()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._switch_to_train(pl_module)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._switch_to_eval(pl_module)
        # see [ NOTE: Why do we use this lazy-switch way to flip param to train mode after eval ]
        self._should_lazy_switch_train = True

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._switch_to_eval(pl_module)
        # see [ NOTE: Why do we use this lazy-switch way to flip param to train mode after eval ]
        self._should_lazy_switch_train = True

    def on_prediction_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._switch_to_eval(pl_module)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._switch_to_eval(pl_module)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        # see [ NOTE: Why do we use this lazy-switch way to flip param to train mode after eval ]
        if self._should_lazy_switch_train:
            self._should_lazy_switch_train = False
            self._switch_to_train(pl_module)
