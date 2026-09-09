"""AI analysis: curated insights always available; live Claude generation when an
ANTHROPIC_API_KEY is present in Streamlit secrets."""
import streamlit as st


def live_available() -> bool:
    try:
        return bool(st.secrets.get("ANTHROPIC_API_KEY", ""))
    except Exception:
        return False


def live_analysis(context: str, question: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
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
        if st.button("Generate live AI analysis", key=f"btn_{key}"):
            with st.spinner("Claude is analyzing ..."):
                try:
                    st.info(live_analysis(context, question))
                except Exception as e:
                    st.warning(f"Live analysis unavailable: {e}")
    else:
        st.caption("Add ANTHROPIC_API_KEY to Streamlit secrets to enable live, "
                   "on-demand AI analysis in addition to the curated insights above.")
