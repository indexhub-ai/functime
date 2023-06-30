from typing import Protocol, Union


class _Stub(Protocol):
    _stub_id: Union[str, None] = None

    @classmethod
    def from_deployed(self, stub_id: str, **kwargs):
        ...

    @property
    def stub_id(self) -> str:
        return self._stub_id

    @property
    def is_fitted(self) -> bool:
        return self._stub_id is not None

    def fit(self, *args, **kwargs):
        ...
