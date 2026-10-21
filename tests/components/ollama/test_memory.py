"""Tests for the Ollama long-term chat memory."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import closing
from pathlib import Path
import re
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, patch

from ollama import EmbedResponse, ResponseError, ShowResponse
import pytest

from homeassistant.components import conversation
from homeassistant.components.ollama.const import (
    CONF_EMBEDDING_MODEL,
    CONF_MAX_HISTORY,
    CONF_MEMORY,
    CONF_MEMORY_LIMIT,
)
from homeassistant.components.ollama.rag.memory import memory_file
from homeassistant.components.ollama.rag.rag_utilities.generate_prompt import (
    MEMORY_HEADER,
)
from homeassistant.config_entries import SOURCE_USER, ConfigSubentry
from homeassistant.const import CONF_MODEL, CONF_NAME
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResult, FlowResultType
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import STORAGE_DIR

from tests.common import MockConfigEntry

AGENT_ID = "conversation.ollama_conversation"
CHAT_MODEL = "test_model:latest"
EMBEDDING_MODEL = "test-embed:latest"
OLD_EMBEDDING_MODEL = "old-embed:latest"
SERVER_MODELS = {"models": [{"model": CHAT_MODEL}, {"model": EMBEDDING_MODEL}]}
VOCABULARY = ["colour", "green", "favourite", "dog", "rex", "light", "kitchen"]
STATUS_LINE = "You have long-term memory of this user."


def fake_vector(text: str) -> list[float]:
    """Embed text as word counts, so texts sharing words are similar."""
    words = re.findall(r"[a-z]+", text.lower())
    return [float(words.count(word)) for word in VOCABULARY] + [0.1]


async def fake_embed(
    model: str, input: str | list[str], **kwargs: Any
) -> EmbedResponse:
    """Embed like an embedding model, and fail like a chat model."""
    if model == CHAT_MODEL:
        raise ResponseError("This server does not support embeddings", 501)
    texts = [input] if isinstance(input, str) else input
    return EmbedResponse(embeddings=[fake_vector(text) for text in texts])


async def fake_show(model: str) -> ShowResponse:
    """Describe the downloaded models like an Ollama server does."""
    capabilities = {CHAT_MODEL: ["completion"], EMBEDDING_MODEL: ["embedding"]}
    return ShowResponse(model_info={}, capabilities=capabilities[model])


async def stream_generator(response: dict) -> AsyncGenerator[dict]:
    """Generate a response from the assistant."""
    yield response


@pytest.fixture(autouse=True)
def memory_dir(hass: HomeAssistant, tmp_path: Path) -> Path:
    """Keep memory files in a temporary config folder."""
    hass.config.config_dir = str(tmp_path)
    (tmp_path / STORAGE_DIR).mkdir()
    return tmp_path / STORAGE_DIR


@pytest.fixture(autouse=True)
def mock_embed() -> Generator[AsyncMock]:
    """Mock the Ollama embedding API."""
    with patch("ollama.AsyncClient.embed", side_effect=fake_embed) as mock:
        yield mock


@pytest.fixture
def mock_config_entry_options() -> dict[str, Any]:
    """Enable memory on the conversation agent."""
    return {
        CONF_MEMORY: True,
        CONF_EMBEDDING_MODEL: EMBEDDING_MODEL,
        CONF_MEMORY_LIMIT: 3,
    }


def conversation_subentry(entry: MockConfigEntry) -> ConfigSubentry:
    """Return the conversation agent of the config entry."""
    return next(
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == "conversation"
    )


def saved_rows(hass: HomeAssistant, subentry_id: str) -> list[tuple[str, str, str]]:
    """Return (user, model, text) of every saved memory, oldest first."""
    with closing(sqlite3.connect(memory_file(hass, subentry_id))) as conn:
        return conn.execute(
            "SELECT user_id, model, content FROM memories ORDER BY id"
        ).fetchall()


async def converse(
    hass: HomeAssistant,
    text: str,
    *,
    user_id: str | None = "user-a",
    conversation_id: str | None = None,
    satellite_id: str | None = None,
    reply: str = "Okay!",
    agent_id: str = AGENT_ID,
) -> tuple[conversation.ConversationResult, str]:
    """Send a message and return the result and the system prompt the AI got."""
    with patch(
        "ollama.AsyncClient.chat",
        side_effect=lambda **kwargs: stream_generator(
            {"message": {"role": "assistant", "content": reply}}
        ),
    ) as mock_chat:
        result = await conversation.async_converse(
            hass,
            text,
            conversation_id,
            Context(user_id=user_id),
            agent_id=agent_id,
            satellite_id=satellite_id,
        )
    return result, mock_chat.call_args.kwargs["messages"][0]["content"]


async def start_memory_flow(hass: HomeAssistant, entry: MockConfigEntry) -> FlowResult:
    """Start adding a conversation agent with memory switched on."""
    flow = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "conversation"), context={"source": SOURCE_USER}
    )
    return await hass.config_entries.subentries.async_configure(
        flow["flow_id"],
        {CONF_MODEL: CHAT_MODEL, CONF_NAME: "Memory agent", CONF_MEMORY: True},
    )


@pytest.mark.usefixtures("mock_init_component")
async def test_memory_recalled_in_later_conversation(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """An exchange from an earlier chat is given to the AI in a new chat."""
    await converse(hass, "My favourite colour is green.", reply="Nice choice!")

    result, prompt = await converse(hass, "What is my favourite colour?")

    assert result.response.speech["plain"]["speech"] == "Okay!"
    assert (
        f"{MEMORY_HEADER}\n- User: My favourite colour is green.\nAssistant: Nice choice!"
        in prompt
    )
    assert "Saved messages: 1 (only their latest 3 are kept" in prompt
    assert saved_rows(hass, conversation_subentry(mock_config_entry).subentry_id) == [
        (
            "user-a",
            EMBEDDING_MODEL,
            "User: My favourite colour is green.\nAssistant: Nice choice!",
        ),
        (
            "user-a",
            EMBEDDING_MODEL,
            "User: What is my favourite colour?\nAssistant: Okay!",
        ),
    ]


@pytest.mark.parametrize(
    "mock_config_entry_options",
    [
        {
            CONF_MEMORY: True,
            CONF_EMBEDDING_MODEL: EMBEDDING_MODEL,
            CONF_MEMORY_LIMIT: 300,
            CONF_MAX_HISTORY: 1,
        }
    ],
)
@pytest.mark.usefixtures("mock_init_component")
async def test_memory_covers_messages_beyond_chat_history(hass: HomeAssistant) -> None:
    """Memory recalls messages of the same chat that left the chat history."""
    first, _ = await converse(hass, "My favourite colour is green.")
    chat = first.conversation_id

    # The AI still sees this message in the chat history, so memory skips it.
    _, prompt = await converse(
        hass, "What is my favourite colour?", conversation_id=chat
    )
    assert MEMORY_HEADER not in prompt

    await converse(hass, "My dog is called Rex.", conversation_id=chat)

    # Now it has left the one-message chat history, so memory brings it back.
    _, prompt = await converse(
        hass, "What is my favourite colour?", conversation_id=chat
    )
    assert "- User: My favourite colour is green." in prompt


@pytest.mark.usefixtures("mock_init_component")
async def test_memory_is_private_per_user(hass: HomeAssistant) -> None:
    """One person never gets another person's memories."""
    await converse(hass, "My favourite colour is green.", user_id="user-a")

    _, prompt = await converse(hass, "What is my favourite colour?", user_id="user-b")

    assert MEMORY_HEADER not in prompt
    assert "Saved messages: 0 " in prompt


@pytest.mark.usefixtures("mock_init_component")
async def test_each_agent_has_its_own_memory(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Two conversation agents keep separate memories."""
    first = conversation_subentry(mock_config_entry)
    second = ConfigSubentry(
        data={
            CONF_MODEL: CHAT_MODEL,
            CONF_MEMORY: True,
            CONF_EMBEDDING_MODEL: EMBEDDING_MODEL,
            CONF_MEMORY_LIMIT: 3,
        },
        subentry_type="conversation",
        title="Second agent",
        unique_id=None,
    )
    with patch("ollama.AsyncClient.list"):
        hass.config_entries.async_add_subentry(mock_config_entry, second)
        await hass.async_block_till_done()
    second_agent = entity_registry.async_get_entity_id(
        "conversation", "ollama", second.subentry_id
    )

    await converse(hass, "My favourite colour is green.")
    _, prompt = await converse(
        hass, "What is my favourite colour?", agent_id=second_agent
    )

    assert MEMORY_HEADER not in prompt
    assert len(saved_rows(hass, first.subentry_id)) == 1
    assert len(saved_rows(hass, second.subentry_id)) == 1


@pytest.mark.usefixtures("mock_init_component")
async def test_memory_keeps_latest_messages_per_person(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Only each person's latest messages are kept, up to the limit."""
    await converse(hass, "Message from B", user_id="user-b")
    for number in range(1, 6):
        await converse(hass, f"Message {number} from A", user_id="user-a")

    rows = saved_rows(hass, conversation_subentry(mock_config_entry).subentry_id)
    assert [text.split("\n")[0] for user, _, text in rows if user == "user-a"] == [
        "User: Message 3 from A",
        "User: Message 4 from A",
        "User: Message 5 from A",
    ]
    assert [text.split("\n")[0] for user, _, text in rows if user == "user-b"] == [
        "User: Message from B"
    ]


@pytest.mark.usefixtures("mock_init_component")
async def test_unrelated_memory_not_sent(hass: HomeAssistant) -> None:
    """Memories scoring below the minimum are not given to the AI."""
    await converse(hass, "My dog is called Rex.")

    _, prompt = await converse(hass, "Turn on the kitchen light")

    assert MEMORY_HEADER not in prompt


@pytest.mark.parametrize(
    ("user_id", "satellite_id"),
    [
        pytest.param("user-a", "assist_satellite.kitchen", id="voice"),
        pytest.param(None, None, id="no-user"),
    ],
)
@pytest.mark.usefixtures("mock_init_component")
async def test_unknown_speakers_not_remembered(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    user_id: str | None,
    satellite_id: str | None,
) -> None:
    """Voice messages and messages without a user are neither used nor saved."""
    _, prompt = await converse(
        hass,
        "My favourite colour is green.",
        user_id=user_id,
        satellite_id=satellite_id,
    )

    assert STATUS_LINE not in prompt
    assert saved_rows(hass, conversation_subentry(mock_config_entry).subentry_id) == []


@pytest.mark.parametrize("mock_config_entry_options", [{}])
@pytest.mark.usefixtures("mock_init_component")
async def test_memory_off_by_default(
    hass: HomeAssistant, memory_dir: Path, mock_embed: AsyncMock
) -> None:
    """Without the memory setting nothing is saved and no file is created."""
    _, prompt = await converse(hass, "My favourite colour is green.")

    assert STATUS_LINE not in prompt
    assert list(memory_dir.iterdir()) == []
    mock_embed.assert_not_called()


@pytest.mark.usefixtures("mock_init_component")
async def test_embedding_failure_keeps_chat_working(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_embed: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the embedding model fails, the chat still answers without memory."""
    mock_embed.side_effect = ResponseError("model not found", 404)

    result, _ = await converse(hass, "My favourite colour is green.")

    assert result.response.speech["plain"]["speech"] == "Okay!"
    assert "Could not embed with test-embed:latest" in caplog.text
    assert saved_rows(hass, conversation_subentry(mock_config_entry).subentry_id) == []


@pytest.mark.usefixtures("mock_init_component")
async def test_settings_turn_memory_on(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_embed: AsyncMock
) -> None:
    """Switching memory on asks for an embedding model and checks that it works."""
    with (
        patch("ollama.AsyncClient.list", return_value=SERVER_MODELS),
        patch("ollama.AsyncClient.show", side_effect=fake_show),
    ):
        result = await start_memory_flow(hass, mock_config_entry)

        assert result["step_id"] == "memory"
        model_field = next(
            value
            for key, value in result["data_schema"].schema.items()
            if key == CONF_EMBEDDING_MODEL
        )
        assert model_field.config["options"] == [
            {"label": f"{EMBEDDING_MODEL} (downloaded)", "value": EMBEDDING_MODEL},
            {"label": "all-minilm", "value": "all-minilm:latest"},
            {"label": "bge-m3", "value": "bge-m3:latest"},
            {"label": "mxbai-embed-large", "value": "mxbai-embed-large:latest"},
            {"label": "nomic-embed-text", "value": "nomic-embed-text:latest"},
            {
                "label": "snowflake-arctic-embed",
                "value": "snowflake-arctic-embed:latest",
            },
        ]

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {CONF_EMBEDDING_MODEL: EMBEDDING_MODEL, CONF_MEMORY_LIMIT: 200},
        )
        await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {
        CONF_MODEL: CHAT_MODEL,
        CONF_MEMORY: True,
        CONF_EMBEDDING_MODEL: EMBEDDING_MODEL,
        CONF_MEMORY_LIMIT: 200,
    }
    mock_embed.assert_called_once_with(model=EMBEDDING_MODEL, input="test")


@pytest.mark.usefixtures("mock_init_component")
async def test_settings_download_missing_model(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """An embedding model that isn't on the server yet is downloaded first."""
    pulled: list[str] = []

    async def delayed_pull(self, model: str) -> None:
        pulled.append(model)
        await asyncio.sleep(0)

    with (
        patch("ollama.AsyncClient.list", return_value=SERVER_MODELS),
        patch("ollama.AsyncClient.show", side_effect=fake_show),
        patch("ollama.AsyncClient.pull", delayed_pull),
    ):
        result = await start_memory_flow(hass, mock_config_entry)
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {CONF_EMBEDDING_MODEL: "mxbai-embed-large", CONF_MEMORY_LIMIT: 300},
        )

        assert result["type"] is FlowResultType.SHOW_PROGRESS
        assert result["step_id"] == "download_embedding"
        await hass.async_block_till_done()

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {}
        )
        await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_EMBEDDING_MODEL] == "mxbai-embed-large:latest"
    assert pulled == ["mxbai-embed-large:latest"]


@pytest.mark.usefixtures("mock_init_component")
async def test_settings_model_that_cannot_embed(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Choosing a chat model instead of an embedding model shows an error."""
    with (
        patch("ollama.AsyncClient.list", return_value=SERVER_MODELS),
        patch("ollama.AsyncClient.show", side_effect=fake_show),
    ):
        result = await start_memory_flow(hass, mock_config_entry)
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {CONF_EMBEDDING_MODEL: CHAT_MODEL, CONF_MEMORY_LIMIT: 300},
        )

        assert result["type"] is FlowResultType.FORM
        assert result["errors"] == {"base": "embedding_failed"}

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {CONF_EMBEDDING_MODEL: EMBEDDING_MODEL, CONF_MEMORY_LIMIT: 300},
        )
        await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_EMBEDDING_MODEL] == EMBEDDING_MODEL


@pytest.mark.parametrize(
    "mock_config_entry_options",
    [
        {
            CONF_MEMORY: True,
            CONF_EMBEDDING_MODEL: OLD_EMBEDDING_MODEL,
            CONF_MEMORY_LIMIT: 300,
        }
    ],
)
@pytest.mark.usefixtures("mock_init_component")
async def test_settings_new_model_updates_saved_messages(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Changing the embedding model re-embeds every saved message."""
    subentry = conversation_subentry(mock_config_entry)
    await converse(hass, "My favourite colour is green.")
    await converse(hass, "My dog is called Rex.")
    assert [model for _, model, _ in saved_rows(hass, subentry.subentry_id)] == [
        OLD_EMBEDDING_MODEL,
        OLD_EMBEDDING_MODEL,
    ]

    with (
        patch("ollama.AsyncClient.list", return_value=SERVER_MODELS),
        patch("ollama.AsyncClient.show", side_effect=fake_show),
    ):
        result = await mock_config_entry.start_subentry_reconfigure_flow(
            hass, subentry.subentry_id
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {CONF_MODEL: CHAT_MODEL, CONF_MEMORY: True}
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {CONF_EMBEDDING_MODEL: EMBEDDING_MODEL, CONF_MEMORY_LIMIT: 300},
        )

        assert result["type"] is FlowResultType.SHOW_PROGRESS
        assert result["step_id"] == "recalculate"
        await hass.async_block_till_done(wait_background_tasks=True)

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {}
        )
        await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert subentry.data[CONF_EMBEDDING_MODEL] == EMBEDDING_MODEL
    assert [model for _, model, _ in saved_rows(hass, subentry.subentry_id)] == [
        EMBEDDING_MODEL,
        EMBEDDING_MODEL,
    ]
