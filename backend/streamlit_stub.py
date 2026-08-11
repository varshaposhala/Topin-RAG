"""Minimal Streamlit stub so app.py can be imported by FastAPI."""

from __future__ import annotations

import types
from typing import Any


class _Secrets(dict):
    def get(self, key, default=None):  # noqa: A003
        return dict.get(self, key, default)


def install_streamlit_stub(secrets: dict | None = None) -> types.ModuleType:
    import sys

    st = types.ModuleType("streamlit")

    def _identity_decorator(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn

        return wrap

    st.cache_data = _identity_decorator
    st.cache_resource = _identity_decorator
    st.secrets = _Secrets(secrets or {})
    st.session_state = {}
    st.set_page_config = lambda **kwargs: None
    st.title = lambda *args, **kwargs: None
    st.markdown = lambda *args, **kwargs: None
    st.error = lambda *args, **kwargs: None
    st.stop = lambda: None
    st.spinner = lambda *args, **kwargs: _NullContext()
    st.chat_message = lambda *args, **kwargs: _NullContext()
    st.chat_input = lambda *args, **kwargs: None
    st.selectbox = lambda *args, **kwargs: None
    st.button = lambda *args, **kwargs: False
    st.download_button = lambda *args, **kwargs: None
    st.number_input = lambda *args, **kwargs: 1
    st.caption = lambda *args, **kwargs: None
    st.info = lambda *args, **kwargs: None
    st.success = lambda *args, **kwargs: None
    st.columns = lambda *args, **kwargs: [_NullContext(), _NullContext()]
    st.container = lambda **kwargs: _NullContext()

    sys.modules["streamlit"] = st
    return st


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None
