from abc import ABC, abstractmethod
from typing import Any

class LowLevelDataFetcherBase(ABC):

    @abstractmethod
    def fetch_dataset(self) -> Any:
        pass

