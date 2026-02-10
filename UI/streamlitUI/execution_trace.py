import streamlit as st

class ExecutionTrace:
    def __init__(self):
        # Persistent data
        if "execution_trace" not in st.session_state:
            st.session_state.execution_trace = []

        if "advanced_logs" not in st.session_state:
            st.session_state.advanced_logs = []

        # Steps container (created once)
        if "trace_steps_container" not in st.session_state:
            st.session_state.trace_steps_container = st.empty()

    def add_step(self, title: str, description: str):
        st.session_state.execution_trace.append({
            "title": title,
            "description": description
        })
        self._render_steps()

    def add_advanced_log(self, message: str):
        st.session_state.advanced_logs.append(message)
        self._render_steps()

    def clear(self):
        st.session_state.execution_trace = []
        st.session_state.advanced_logs = []
        self._render_steps()

    def _render_steps(self):
        with st.session_state.trace_steps_container.container():
            for step in st.session_state.execution_trace:
                st.markdown(f"**{step['title']}**")
                st.markdown(f"• {step['description']}")

            if st.session_state.advanced_logs:
                with st.expander("Advanced execution details"):
                    for log in st.session_state.advanced_logs:
                        st.caption(log)
