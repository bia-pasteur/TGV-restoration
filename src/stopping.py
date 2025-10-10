from typing import List

from tgv.descent.pdhg import StoppingCriterion


class ProportionPlateauStoppingCriterion(StoppingCriterion):
    """Stops the descent if it stopped improving for n updates

    Attributes:
        ema (float): Exponential moving average to smooth the loss
        proportion (float): Should reduce loss by proportion in n updates
        n (int): Number of updates to consider

    """

    def __init__(self, proportion: float, n: int, ema=0.9) -> None:
        super().__init__()
        self.ema = ema
        self.proportion = proportion
        self.n = n
        self.last_values: List[float] = []
        # Let's store all losses for analysis
        self._losses: List[float] = []

    def update(self, _u, _v, loss: float) -> bool:
        self._losses.append(loss)

        if not self.last_values:
            self.last_values.append(loss)
        else:
            self.last_values.append(self.last_values[-1] * self.ema + loss * (1 - self.ema))

        if len(self.last_values) > self.n:
            old_value = self.last_values.pop(0)

            if 1 - self.last_values[-1] / old_value < self.proportion:
                return True

        return False

    def reset(self):
        self.last_values = []
