"""Config flow for Ollama integration."""

import asyncio
from collections.abc import Mapping
import logging
import sys
from typing import Any, override

import httpx
import ollama
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_NAME,
    CONF_PROMPT,
    CONF_URL,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.util.ssl import get_default_context

from . import OllamaConfigEntry
from .const import (
    CONF_EMBEDDING_MODEL,
    CONF_KEEP_ALIVE,
    CONF_MAX_HISTORY,
    CONF_MEMORY,
    CONF_MEMORY_LIMIT,
    CONF_NUM_CTX,
    CONF_THINK,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_THINK,
    DEFAULT_TIMEOUT,
    DOMAIN,
    EMBEDDING_MODEL_NAMES,
    MAX_MEMORY_LIMIT,
    MIN_MEMORY_LIMIT,
    MIN_NUM_CTX,
    MODEL_NAMES,
)
from .rag.memory import ChatMemory

_LOGGER = logging.getLogger(__name__)

# Loading the embedding model for the first time can take a while.
EMBEDDING_CHECK_TIMEOUT = 60


STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    },
)

STEP_REAUTH_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)


class OllamaConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Ollama."""

    VERSION = 3
    MINOR_VERSION = 3

    async def _async_validate_connection(
        self, url: str, api_key: str | None
    ) -> dict[str, str]:
        """Validate connection and credentials against the Ollama server."""
        errors: dict[str, str] = {}

        try:
            client = ollama.AsyncClient(
                host=url,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
                verify=get_default_context(),
            )

            async with asyncio.timeout(DEFAULT_TIMEOUT):
                await client.list()

        except ollama.ResponseError as err:
            if err.status_code in (401, 403):
                errors["base"] = "invalid_auth"
            else:
                _LOGGER.warning(
                    "Error response from Ollama server at %s: status %s, detail: %s",
                    url,
                    err.status_code,
                    str(err),
                )
                errors["base"] = "unknown"
        except TimeoutError, ConnectionError:
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"

        return errors

    @override
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}
        url = user_input[CONF_URL].strip()
        api_key = user_input.get(CONF_API_KEY)
        if api_key:
            api_key = api_key.strip()

        try:
            url = cv.url(url)
        except vol.Invalid:
            errors["base"] = "invalid_url"
            return self.async_show_form(
                step_id="user",
                data_schema=self.add_suggested_values_to_schema(
                    STEP_USER_DATA_SCHEMA, user_input
                ),
                errors=errors,
            )

        self._async_abort_entries_match({CONF_URL: url})
        errors = await self._async_validate_connection(url, api_key)

        if errors:
            return self.async_show_form(
                step_id="user",
                data_schema=self.add_suggested_values_to_schema(
                    STEP_USER_DATA_SCHEMA, user_input
                ),
                errors=errors,
            )

        entry_data: dict[str, str] = {CONF_URL: url}
        if api_key:
            entry_data[CONF_API_KEY] = api_key

        return self.async_create_entry(title=url, data=entry_data)

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle reauthentication when existing credentials are invalid."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reauthentication confirmation."""
        reauth_entry = self._get_reauth_entry()

        if user_input is None:
            return self.async_show_form(
                step_id="reauth_confirm",
                data_schema=STEP_REAUTH_DATA_SCHEMA,
            )

        api_key = user_input.get(CONF_API_KEY)
        if api_key:
            api_key = api_key.strip()

        errors = await self._async_validate_connection(
            reauth_entry.data[CONF_URL], api_key
        )
        if errors:
            return self.async_show_form(
                step_id="reauth_confirm",
                data_schema=self.add_suggested_values_to_schema(
                    STEP_REAUTH_DATA_SCHEMA, user_input
                ),
                errors=errors,
            )

        updated_data = {
            **reauth_entry.data,
            CONF_URL: reauth_entry.data[CONF_URL],
        }
        if api_key:
            updated_data[CONF_API_KEY] = api_key
        else:
            updated_data.pop(CONF_API_KEY, None)

        updated_options = {
            key: value
            for key, value in reauth_entry.options.items()
            if key != CONF_API_KEY
        }

        return self.async_update_reload_and_abort(
            reauth_entry,
            data=updated_data,
            options=updated_options,
        )

    @classmethod
    @callback
    @override
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": OllamaConversationSubentryFlowHandler,
            "ai_task_data": OllamaSubentryFlowHandler,
        }


class OllamaSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing Ollama subentries."""

    def __init__(self) -> None:
        """Initialize the subentry flow."""
        super().__init__()
        self._name: str | None = None
        self._model: str | None = None
        self.download_task: asyncio.Task | None = None
        self._config_data: dict[str, Any] | None = None

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    @property
    def _client(self) -> ollama.AsyncClient:
        """Return the Ollama client."""
        entry: OllamaConfigEntry = self._get_entry()
        return entry.runtime_data

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle model selection and configuration step."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is None:
            # Get available models from Ollama server
            try:
                async with asyncio.timeout(DEFAULT_TIMEOUT):
                    response = await self._client.list()

                downloaded_models: set[str] = {
                    model_info["model"] for model_info in response.get("models", [])
                }
            except TimeoutError, httpx.HTTPError, ConnectionError:
                _LOGGER.exception("Failed to get models from Ollama server")
                return self.async_abort(reason="cannot_connect")

            # Show models that have been downloaded first, followed by all known
            # models (only latest tags).
            models_to_list = [
                SelectOptionDict(label=f"{m} (downloaded)", value=m)
                for m in sorted(downloaded_models)
            ] + [
                SelectOptionDict(label=m, value=f"{m}:latest")
                for m in sorted(MODEL_NAMES)
                if m not in downloaded_models
            ]

            if self._is_new:
                options = {}
            else:
                options = self._get_reconfigure_subentry().data.copy()

            return self.async_show_form(
                step_id="set_options",
                data_schema=vol.Schema(
                    ollama_config_option_schema(
                        self.hass,
                        self._is_new,
                        self._subentry_type,
                        options,
                        models_to_list,
                    )
                ),
            )

        self._model = user_input[CONF_MODEL]
        if self._is_new:
            self._name = user_input.pop(CONF_NAME)
        self._config_data = user_input

        # Check if model needs to be downloaded
        try:
            async with asyncio.timeout(DEFAULT_TIMEOUT):
                response = await self._client.list()

            currently_downloaded_models: set[str] = {
                model_info["model"] for model_info in response.get("models", [])
            }

            if self._model not in currently_downloaded_models:
                # Ollama server needs to download model first
                return await self.async_step_download()
        except Exception:
            _LOGGER.exception("Failed to check model availability")
            return self.async_abort(reason="cannot_connect")

        return await self.async_step_finish()

    async def async_step_download(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Step to wait for Ollama server to download a model."""
        assert self._model is not None

        if self.download_task is None:
            # Tell Ollama server to pull the model.
            # The task will block until the model and metadata are fully
            # downloaded.
            self.download_task = self.hass.async_create_background_task(
                self._client.pull(self._model),
                f"Downloading {self._model}",
            )

        if self.download_task.done():
            if err := self.download_task.exception():
                _LOGGER.exception("Unexpected error while downloading model: %s", err)
                return self.async_show_progress_done(next_step_id="failed")

            return self.async_show_progress_done(next_step_id="finish")

        return self.async_show_progress(
            step_id="download",
            progress_action="download",
            progress_task=self.download_task,
        )

    async def async_step_failed(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Step after model downloading has failed."""
        return self.async_abort(reason="download_failed")

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Save the agent once its models are ready."""
        assert self._config_data is not None
        if self._is_new:
            return self.async_create_entry(
                title=self._name,
                data=self._config_data,
            )
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            data=self._config_data,
        )

    async_step_user = async_step_set_options
    async_step_reconfigure = async_step_set_options


class OllamaConversationSubentryFlowHandler(OllamaSubentryFlowHandler):
    """Flow for managing Ollama conversation agents, which can have memory."""

    def __init__(self) -> None:
        """Initialize the subentry flow."""
        super().__init__()
        self._embedding_model: str | None = None
        self._embedding_download_task: asyncio.Task | None = None
        self._memory_ready = False
        self._memory_update: ChatMemory | None = None
        self._recalculate_task: asyncio.Task[int] | None = None
        self._memory_errors: dict[str, str] | None = None

    @override
    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle model selection and configuration step."""
        if user_input is not None and not self._is_new:
            # Keep the memory choices for when memory is switched back on.
            previous = self._get_reconfigure_subentry().data
            for key in (CONF_EMBEDDING_MODEL, CONF_MEMORY_LIMIT):
                if key in previous:
                    user_input.setdefault(key, previous[key])
        return await super().async_step_set_options(user_input)

    async def async_step_memory(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Choose the embedding model and how many messages to remember."""
        assert self._config_data is not None
        # Coming back here after a progress step passes an empty input.
        if not user_input:
            errors, self._memory_errors = self._memory_errors, None
            return await self._async_show_memory_form(errors)

        self._embedding_model = _with_tag(user_input[CONF_EMBEDDING_MODEL])
        self._embedding_download_task = None
        self._recalculate_task = None
        self._config_data[CONF_EMBEDDING_MODEL] = self._embedding_model
        self._config_data[CONF_MEMORY_LIMIT] = int(user_input[CONF_MEMORY_LIMIT])
        try:
            async with asyncio.timeout(DEFAULT_TIMEOUT):
                response = await self._client.list()
        except TimeoutError, httpx.HTTPError, ConnectionError:
            _LOGGER.exception(
                "Failed to check the embedding model on the Ollama server"
            )
            return self.async_abort(reason="cannot_connect")

        if self._embedding_model not in {
            model_info["model"] for model_info in response.get("models", [])
        }:
            return await self.async_step_download_embedding()
        return await self.async_step_check_embedding()

    async def _async_show_memory_form(
        self, errors: dict[str, str] | None = None
    ) -> SubentryFlowResult:
        """Show the memory form with the embedding models to choose from."""
        assert self._config_data is not None
        try:
            models_to_list = await self._async_embedding_models()
        except TimeoutError, httpx.HTTPError, ConnectionError:
            _LOGGER.exception("Failed to list embedding models on the Ollama server")
            return self.async_abort(reason="cannot_connect")

        return self.async_show_form(
            step_id="memory",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_EMBEDDING_MODEL,
                        description={
                            "suggested_value": self._config_data.get(
                                CONF_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL
                            )
                        },
                    ): SelectSelector(
                        SelectSelectorConfig(options=models_to_list, custom_value=True)
                    ),
                    vol.Required(
                        CONF_MEMORY_LIMIT,
                        description={
                            "suggested_value": self._config_data.get(
                                CONF_MEMORY_LIMIT, DEFAULT_MEMORY_LIMIT
                            )
                        },
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=MIN_MEMORY_LIMIT,
                            max=MAX_MEMORY_LIMIT,
                            step=1,
                            mode=NumberSelectorMode.BOX,
                        )
                    ),
                }
            ),
            errors=errors,
        )

    async def _async_embedding_models(self) -> list[SelectOptionDict]:
        """List downloaded embedding models first, then known ones to download."""
        async with asyncio.timeout(DEFAULT_TIMEOUT):
            response = await self._client.list()
            names = sorted(
                model_info["model"] for model_info in response.get("models", [])
            )
            details = await asyncio.gather(
                *(self._client.show(name) for name in names), return_exceptions=True
            )
        downloaded = [
            name
            for name, info in zip(names, details, strict=True)
            if _is_embedding_model(name, info)
        ]
        return [
            SelectOptionDict(label=f"{name} (downloaded)", value=name)
            for name in downloaded
        ] + [
            SelectOptionDict(label=name, value=f"{name}:latest")
            for name in EMBEDDING_MODEL_NAMES
            if f"{name}:latest" not in downloaded
        ]

    async def async_step_download_embedding(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Step to wait for Ollama server to download the embedding model."""
        assert self._embedding_model is not None

        if self._embedding_download_task is None:
            self._embedding_download_task = self.hass.async_create_background_task(
                self._client.pull(self._embedding_model),
                f"Downloading {self._embedding_model}",
            )

        if not self._embedding_download_task.done():
            return self.async_show_progress(
                step_id="download_embedding",
                progress_action="download_embedding",
                progress_task=self._embedding_download_task,
            )

        try:
            await self._embedding_download_task
        except Exception:
            _LOGGER.exception("Unexpected error while downloading model")
            return self.async_show_progress_done(next_step_id="failed")
        return self.async_show_progress_done(next_step_id="check_embedding")

    async def async_step_check_embedding(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Check that the chosen model can create embeddings."""
        assert self._config_data is not None
        assert self._embedding_model is not None
        try:
            async with asyncio.timeout(EMBEDDING_CHECK_TIMEOUT):
                await self._client.embed(model=self._embedding_model, input="test")
        except ollama.RequestError, ollama.ResponseError:
            return await self._async_show_memory_form({"base": "embedding_failed"})
        except TimeoutError, httpx.HTTPError, ConnectionError:
            return await self._async_show_memory_form({"base": "cannot_connect"})
        self._memory_ready = True

        if not self._is_new:
            memory = ChatMemory(
                self.hass,
                self._client,
                self._get_reconfigure_subentry().subentry_id,
                self._embedding_model,
                self._config_data[CONF_MEMORY_LIMIT],
            )
            if await memory.async_count_stale():
                self._memory_update = memory
                return await self.async_step_recalculate()
        return await self.async_step_finish()

    async def async_step_recalculate(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Step to wait while saved messages are embedded with the new model."""
        assert self._memory_update is not None

        if self._recalculate_task is None:
            self._recalculate_task = self.hass.async_create_background_task(
                self._memory_update.async_recalculate(self.async_update_progress),
                "Updating Ollama memories",
            )

        if not self._recalculate_task.done():
            return self.async_show_progress(
                step_id="recalculate",
                progress_action="recalculate",
                progress_task=self._recalculate_task,
            )

        try:
            await self._recalculate_task
        except Exception:
            _LOGGER.exception("Could not update the saved messages")
            # Back to the memory screen: nothing is saved and the user can retry.
            self._memory_errors = {"base": "recalculate_failed"}
            self._memory_ready = False
            return self.async_show_progress_done(next_step_id="memory")
        return self.async_show_progress_done(next_step_id="finish")

    @override
    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Ask for the memory settings first when memory is switched on."""
        assert self._config_data is not None
        if self._config_data.get(CONF_MEMORY) and not self._memory_ready:
            return await self.async_step_memory()
        return await super().async_step_finish(user_input)

    async_step_user = async_step_set_options
    async_step_reconfigure = async_step_set_options


def _with_tag(model: str) -> str:
    """Return the model name with a tag, as Ollama lists downloaded models."""
    return model if ":" in model else f"{model}:latest"


def _is_embedding_model(name: str, info: ollama.ShowResponse | BaseException) -> bool:
    """Return if a downloaded model can create embeddings."""
    if isinstance(info, ollama.ShowResponse) and info.capabilities:
        return "embedding" in info.capabilities
    # Older Ollama servers don't report capabilities.
    return name.partition(":")[0] in EMBEDDING_MODEL_NAMES


def filter_invalid_llm_apis(hass: HomeAssistant, selected_apis: list[str]) -> list[str]:
    """Filter a list of LLM API IDs against those available."""

    valid_llm_apis = [api.id for api in llm.async_get_apis(hass)]

    return [api for api in selected_apis if api in valid_llm_apis]


def ollama_config_option_schema(
    hass: HomeAssistant,
    is_new: bool,
    subentry_type: str,
    options: Mapping[str, Any],
    models_to_list: list[SelectOptionDict],
) -> dict:
    """Ollama options schema."""
    if is_new:
        if subentry_type == "ai_task_data":
            default_name = DEFAULT_AI_TASK_NAME
        else:
            default_name = DEFAULT_CONVERSATION_NAME

        schema: dict = {
            # Name field is no longer allowed in config flow schemas
            # pylint: disable-next=home-assistant-config-flow-name-field
            vol.Required(CONF_NAME, default=default_name): str,
        }
    else:
        schema = {}

    selected_llm_apis = filter_invalid_llm_apis(
        hass, options.get(CONF_LLM_HASS_API, [])
    )

    schema.update(
        {
            vol.Required(
                CONF_MODEL,
                description={"suggested_value": options.get(CONF_MODEL, DEFAULT_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(options=models_to_list, custom_value=True)
            ),
        }
    )
    if subentry_type == "conversation":
        schema.update(
            {
                vol.Optional(
                    CONF_PROMPT,
                    description={
                        "suggested_value": options.get(
                            CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                        )
                    },
                ): TemplateSelector(),
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={"suggested_value": selected_llm_apis},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(
                                label=api.name,
                                value=api.id,
                            )
                            for api in llm.async_get_apis(hass)
                        ],
                        multiple=True,
                    )
                ),
            }
        )
    schema.update(
        {
            vol.Optional(
                CONF_NUM_CTX,
                description={
                    "suggested_value": options.get(CONF_NUM_CTX, DEFAULT_NUM_CTX)
                },
            ): NumberSelector(
                NumberSelectorConfig(
                    min=MIN_NUM_CTX,
                    step=1,
                    mode=NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_MAX_HISTORY,
                description={
                    "suggested_value": options.get(
                        CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY
                    )
                },
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=sys.maxsize, step=1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_KEEP_ALIVE,
                description={
                    "suggested_value": options.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE)
                },
            ): NumberSelector(
                NumberSelectorConfig(
                    min=-1, max=sys.maxsize, step=1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_THINK,
                description={
                    "suggested_value": options.get("think", DEFAULT_THINK),
                },
            ): BooleanSelector(),
        }
    )
    if subentry_type == "conversation":
        schema[
            vol.Optional(
                CONF_MEMORY,
                description={"suggested_value": options.get(CONF_MEMORY, False)},
            )
        ] = BooleanSelector()

    return schema
