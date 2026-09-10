"""AI analysis: curated insights always shown; live Claude generation when a key is
provided in the sidebar or in Streamlit secrets."""
import streamlit as st


def _key() -> str:
    k = st.session_state.get("claude_api_key", "")
    if k:
        return k
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return ""


def live_available() -> bool:
    return bool(_key())


def live_analysis(context: str, question: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_key())
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        messages=[{"role": "user", "content":
                   f"You are a pricing analytics expert reviewing a Ford Explorer "
                   f"yield management dashboard. Data and results:\n{context}\n\n"
                   f"Task: {question}\nRespond in concise, formal business prose, "
                   f"under 250 words, no headers."}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def ai_section(key: str, curated: str, context: str, question: str):
    """Render the AI analysis block used at the end of each tab."""
    st.markdown("#### AI analysis")
    st.markdown(curated)
    if live_available():
        if st.button("Generate live AI analysis with Claude", key=f"btn_{key}"):
            with st.spinner("Claude is analyzing ..."):
                try:
                    st.info(live_analysis(context, question))
                except Exception as e:
                    st.warning(f"Live analysis unavailable: {e}")
    else:
        st.caption("Enter a Claude API key in the sidebar to generate live, on-demand "
                   "AI analysis for this tab.")
