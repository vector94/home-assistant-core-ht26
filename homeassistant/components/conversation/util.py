"""Utility functions for conversation integration."""

import logging

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    intent,
    llm,
)
from homeassistant.helpers.typing import UNDEFINED, UndefinedType

from .chat_log import AssistantContent, ChatLog, ToolResultContent
from .const import DOMAIN
from .models import ConversationInput, ConversationResult

_LOGGER = logging.getLogger(__name__)


@callback
def async_get_result_from_chat_log(
    user_input: ConversationInput, chat_log: ChatLog
) -> ConversationResult:
    """Get the result from the chat log."""
    tool_results = [
        content.tool_result
        for content in chat_log.content[chat_log.llm_input_provided_index :]
        if isinstance(content, ToolResultContent)
        and isinstance(content.tool_result, llm.IntentResponseDict)
    ]

    if tool_results:
        intent_response = tool_results[-1].original
    else:
        intent_response = intent.IntentResponse(language=user_input.language)

    if not isinstance((last_content := chat_log.content[-1]), AssistantContent):
        _LOGGER.error(
            "Last content in chat log is not an AssistantContent: %s."
            " This could be due to the model not returning a valid response",
            last_content,
        )
        raise HomeAssistantError("Unable to get response")

    intent_response.async_set_speech(last_content.content or "")

    return ConversationResult(
        response=intent_response,
        conversation_id=chat_log.conversation_id,
        continue_conversation=chat_log.continue_conversation,
    )


@callback
def async_move_agent_to_subentry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    parent_entry: ConfigEntry,
    subentry: ConfigSubentry,
    all_disabled: bool,
) -> None:
    """Move the conversation entity and device of an entry to a subentry.

    Used by integrations that migrate from one config entry per agent
    to one parent entry with one subentry per agent.
    """
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)

    conversation_entity_id = entity_registry.async_get_entity_id(
        DOMAIN,
        entry.domain,
        entry.entry_id,
    )
    device = device_registry.async_get_device_by_identifier(
        (entry.domain, entry.entry_id), entry.entry_id
    )

    if conversation_entity_id is not None:
        conversation_entity_entry = entity_registry.entities[conversation_entity_id]
        entity_disabled_by = conversation_entity_entry.disabled_by
        if (
            entity_disabled_by is er.RegistryEntryDisabler.CONFIG_ENTRY
            and not all_disabled
        ):
            # Device and entity registries will set the disabled_by flag to None
            # when moving a device or entity disabled by CONFIG_ENTRY to an enabled
            # config entry, but we want to set it to DEVICE or USER instead,
            entity_disabled_by = (
                er.RegistryEntryDisabler.DEVICE
                if device
                else er.RegistryEntryDisabler.USER
            )
        entity_registry.async_update_entity(
            conversation_entity_id,
            config_entry_id=parent_entry.entry_id,
            config_subentry_id=subentry.subentry_id,
            disabled_by=entity_disabled_by,
            new_unique_id=subentry.subentry_id,
        )

    if device is not None:
        # Device and entity registries will set the disabled_by flag to None
        # when moving a device or entity disabled by CONFIG_ENTRY to an enabled
        # config entry, but we want to set it to USER instead,
        device_disabled_by: dr.DeviceEntryDisabler | UndefinedType = UNDEFINED
        if (
            device.disabled_by is dr.DeviceEntryDisabler.CONFIG_ENTRY
            and not all_disabled
        ):
            device_disabled_by = dr.DeviceEntryDisabler.USER
        device_registry.async_update_device(
            device.id,
            disabled_by=device_disabled_by,
            new_identifiers={(entry.domain, subentry.subentry_id)},
            new_config_entry_id=parent_entry.entry_id,
            new_config_subentry_id=subentry.subentry_id,
        )
