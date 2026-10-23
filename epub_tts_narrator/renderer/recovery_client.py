"""Bounded output-budget retries for the renderer's small analysis decisions."""

from copy import deepcopy
import logging

from epub_tts_narrator.provider import OutputTruncatedError


_LOGGER = logging.getLogger(__name__)


class RecoveryModelClient:
    """Retry only truncated decisions, without rerunning synthesis or prior votes.

    Reasoning models spend output tokens on reasoning as well as JSON. Start
    with room for both and grow to the user's configured ceiling when needed.
    Each call owns its retry clients, so concurrent decisions cannot change one
    another's settings or cache identity.
    """

    def __init__(self, settings):
        from epub_tts_narrator.provider import OpenAICompatibleClient

        self._options = dict(settings)
        maximum = self._options.get("max_output_tokens")
        if maximum is None:
            maximum = 32768
        # Validate the supplied ceiling before using it to calculate budgets.
        client = OpenAICompatibleClient(**{**self._options, "max_output_tokens": maximum})
        self._budgets = tuple(dict.fromkeys((min(8192, maximum), min(32768, maximum), maximum)))
        self._identity = {**client.identity, "output_budget_retry": {
            "version": 1, "budgets": list(self._budgets),
        }}

    @property
    def identity(self):
        return deepcopy(self._identity)

    def complete(self, system, payload):
        from epub_tts_narrator.provider import OpenAICompatibleClient

        for index, budget in enumerate(self._budgets):
            client = OpenAICompatibleClient(**{**self._options, "max_output_tokens": budget})
            try:
                return client.complete(system, deepcopy(payload))
            except OutputTruncatedError:
                if index + 1 == len(self._budgets):
                    raise OutputTruncatedError(
                        "The speech recovery model exhausted its output budget "
                        f"after {index + 1} request(s), reaching {budget} tokens. "
                        "The audio is saved. Increase provider.max_output_tokens "
                        "or reduce provider.reasoning_effort in the config file, then rerun."
                    ) from None
                _LOGGER.warning(
                    "Speech recovery response reached %d output tokens; retrying the "
                    "same decision with %d tokens. Generated audio is retained.",
                    budget, self._budgets[index + 1],
                )
        raise AssertionError("Unreachable output-budget retry state")
