# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from __future__ import annotations

import torch

__all__ = ["Rotation90Generator"]

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _common_param_check, _range_bound


class Rotation90Generator(RandomGeneratorBase):
    """Generate uniformly sampled integer quarter-turn counts."""

    def __init__(self, times: tuple[int, int]) -> None:
        super().__init__()
        self.register_buffer(
            "times",
            _range_bound(times, "times", center=0.0, bounds=(-3.0, 3.0)),
        )

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        bounds = _range_bound(self.times, "times", center=0.0, bounds=(-3.0, 3.0), device=device, dtype=dtype)

        if not torch.isfinite(bounds).all():
            raise ValueError("times must contain finite values.")

        if not torch.equal(bounds, bounds.round()):
            raise ValueError("times must contain whole-number quarter turns.")

        self.low = int(bounds[0].item())
        self.high = int(bounds[1].item())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.make_samplers(self.device, self.dtype)

    def forward(
        self,
        batch_shape: tuple[int, ...],
        same_on_batch: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)

        sample_count = 1 if same_on_batch and batch_size > 0 else batch_size

        times = torch.randint(
            low=self.low,
            high=self.high + 1,
            size=(sample_count,),
            device=self.device,
        )

        if same_on_batch:
            times = times.expand(batch_size)
        return {"times": times}
