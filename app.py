import os

import streamlit as st
from dotenv import load_dotenv

try:
    from anthropic import (
        Anthropic,
        APIConnectionError,
        APIError,
        AuthenticationError,
    )
except ImportError:
    st.error("Missing anthropic SDK. Run: pip install -r requirements.txt")
    st.stop()

load_dotenv()

FALLBACK_MODELS = ["claude-haiku-5-5", "claude-sonnet-5", "claude-opus-5"]
EFFORT_OPTIONS = ["unspecified", "low", "medium", "high", "max"]
THINKING_ADAPTIVE = "Adaptive (uses effort above)"
THINKING_MANUAL = "Manual token budget"


class ClaudeChatService:
    def __init__(self, api_key):
        self.api_key = api_key
        self._client = Anthropic(api_key=api_key) if api_key else None

    @property
    def is_configured(self):
        return self._client is not None

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_model_ids(api_key):
        try:
            client = Anthropic(api_key=api_key)
            return [model.id for model in client.models.list()]
        except Exception:
            return []

    def list_models(self):
        if not self.is_configured:
            return []
        return self._fetch_model_ids(self.api_key)

    def send(self, request):
        return ChatResponse(self._client.messages.create(**request.to_params()))


class ChatRequest:
    def __init__(
        self,
        messages,
        model,
        max_tokens,
        system_prompt,
        effort,
        thinking_enabled,
        thinking_mode,
        budget_tokens,
    ):
        self.messages = messages
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.effort = effort
        self.thinking_enabled = thinking_enabled
        self.thinking_mode = thinking_mode
        self.budget_tokens = budget_tokens

    def to_params(self):
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
        }

        if self.system_prompt:
            params["system"] = self.system_prompt

        if self.thinking_enabled:
            if self.thinking_mode == THINKING_MANUAL and self.budget_tokens:
                params["thinking"] = {"type": "enabled", "budget_tokens": int(self.budget_tokens)}
            else:
                thinking_config = {"type": "adaptive"}
                if self.effort != "unspecified":
                    thinking_config["effort"] = self.effort
                params["thinking"] = thinking_config
        elif self.effort != "unspecified":
            params["output_config"] = {"effort": self.effort}

        return params


class ChatResponse:
    def __init__(self, raw_response):
        self._raw = raw_response

    @property
    def text(self):
        blocks = [block.text for block in self._raw.content if getattr(block, "type", None) == "text"]
        return "".join(blocks)

    @property
    def usage(self):
        return getattr(self._raw, "usage", None)


class ConversationMemory:
    def __init__(self, state):
        self._state = state
        if "messages" not in self._state:
            self._state.messages = []

    @property
    def messages(self):
        return self._state.messages

    def add(self, role, content):
        self._state.messages.append({"role": role, "content": content})

    def clear(self):
        self._state.messages = []

    def as_api_messages(self):
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]


class SidebarConfig:
    def __init__(self, service):
        self.service = service
        self.model = None
        self.max_tokens = None
        self.effort = None
        self.thinking_enabled = False
        self.thinking_mode = None
        self.budget_tokens = None
        self.system_prompt = None
        self.clear_requested = False

    def render(self):
        with st.sidebar:
            st.header("Configuration")

            if not self.service.is_configured:
                st.error("ANTHROPIC_API_KEY not found. Copy .env.example to .env and set your key.")

            available_models = self.service.list_models()
            model_options = available_models or FALLBACK_MODELS
            if self.service.is_configured and not available_models:
                st.caption("Could not list account models; showing fallback list.")

            self.model = st.selectbox("Model", options=model_options, index=0)
            self.max_tokens = st.slider("Max tokens", min_value=256, max_value=4096, value=1024, step=128)

            self.effort = st.selectbox(
                "Effort level",
                options=EFFORT_OPTIONS,
                index=0,
                help=(
                    "Reasoning effort level. Sent inside 'thinking' when adaptive "
                    "Thinking is active, otherwise inside 'output_config'."
                ),
            )

            st.divider()
            self.thinking_enabled = st.checkbox("Enable Thinking (extended reasoning)")

            if self.thinking_enabled:
                self.thinking_mode = st.radio("Thinking mode", options=[THINKING_ADAPTIVE, THINKING_MANUAL])
                if self.thinking_mode == THINKING_MANUAL:
                    upper_bound = max(self.max_tokens - 1, 1024)
                    self.budget_tokens = st.number_input(
                        "budget_tokens",
                        min_value=1024,
                        max_value=upper_bound,
                        value=min(2048, upper_bound),
                        step=256,
                    )

            st.divider()
            self.system_prompt = st.text_area(
                "System prompt",
                value="You are a helpful and concise assistant.",
                height=120,
            )

            st.divider()
            self.clear_requested = st.button("Clear conversation")

        return self


class ChatApp:
    def __init__(self):
        st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide")
        self.service = ClaudeChatService(os.getenv("ANTHROPIC_API_KEY"))
        self.memory = ConversationMemory(st.session_state)
        self.config = SidebarConfig(self.service)

    def render_history(self):
        for message in self.memory.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def build_request(self, user_input):
        self.memory.add("user", user_input)
        system_prompt = self.config.system_prompt.strip() if self.config.system_prompt else None
        return ChatRequest(
            messages=self.memory.as_api_messages(),
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system_prompt=system_prompt,
            effort=self.config.effort,
            thinking_enabled=self.config.thinking_enabled,
            thinking_mode=self.config.thinking_mode,
            budget_tokens=self.config.budget_tokens,
        )

    def handle_input(self):
        user_input = st.chat_input("Type your message...")
        if not user_input:
            return

        if not self.service.is_configured:
            st.error("Set ANTHROPIC_API_KEY in your .env file before chatting.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(user_input)

        request = self.build_request(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")
            try:
                response = self.service.send(request)
                answer = response.text or "_(No text content returned.)_"
                placeholder.markdown(answer)

                if response.usage is not None:
                    st.caption(
                        f"input tokens: {response.usage.input_tokens} - output tokens: {response.usage.output_tokens}"
                    )

                self.memory.add("assistant", answer)

            except AuthenticationError:
                placeholder.error("Invalid API key. Check ANTHROPIC_API_KEY in your .env file.")
            except APIConnectionError:
                placeholder.error("Could not connect to the Anthropic API. Check your internet connection.")
            except APIError as error:
                placeholder.error(f"Anthropic API error: {error}")
            except Exception as error:
                placeholder.error(f"Unexpected error: {error}")

    def run(self):
        self.config.render()

        if self.config.clear_requested:
            self.memory.clear()
            st.rerun()

        st.title("Claude Chat")
        st.caption("Day 10 practical task - TCS FDE Training (Claude Architect Program)")

        self.render_history()
        self.handle_input()


if __name__ == "__main__":
    ChatApp().run()
