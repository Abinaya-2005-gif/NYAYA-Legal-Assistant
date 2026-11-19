
import streamlit as st
from agents import run

# PAGE CONFIGURATION
st.set_page_config(
    page_title="NYAYA - Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# HEADER
st.title("⚖️ NYAYA")

st.caption(
    "AI-Powered Situation-Based Legal Assistance "
    "| English | தமிழ் | हिन्दी"
)

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR
with st.sidebar:

    st.header("⚙️ Settings")

    lang = st.selectbox(
        "Answer Language",
        ["Auto", "English", "Tamil", "Hindi"]
    )

    mode = st.radio(
        "Select Pipeline",
        ["agentic", "basic", "plain"],
        format_func={
            "agentic": "Agentic RAG",
            "basic": "Basic RAG",
            "plain": "Plain LLM"
        }.get
    )

    st.warning(
        "NYAYA provides general legal information, "
        "not professional legal advice."
    )

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# WELCOME MESSAGE
if not st.session_state.messages:

    st.info("""
    👋 Welcome to NYAYA!

    Describe any legal problem in your daily life.

    You don't need to know the Act name or Section number.

    NYAYA will understand your situation and ask
    necessary follow-up questions before providing guidance.
    """)

# DISPLAY CHAT HISTORY
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("docs"):

                with st.expander("📚 Legal Sources"):

                    for doc in message["docs"]:

                        st.markdown(
                            f"**{doc['meta']['act']} - "
                            f"{doc['meta']['section']}**"
                        )

                        st.text(doc["text"][:800])

            if message.get("trace"):

                with st.expander("🤖 Agent Execution Trace"):

                    for step in message["trace"]:
                        st.write("•", step)

# CHAT INPUT
question = st.chat_input(
    "Describe your legal situation..."
)

# PROCESS USER QUESTION
if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.messages[:-1]

    conversation = ""

    for msg in history[-10:]:

        role = msg["role"].upper()

        conversation += f"\n{role}: {msg['content']}\n"

    # GENERATE RESPONSE
    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "NYAYA agents are analysing your situation..."
            ):

                result = run(
                    question,
                    mode=mode,
                    forced_language=None if lang == "Auto" else lang,
                    conversation_history=conversation
                )

            st.info(f"Pipeline: {result['mode']}")

            st.markdown(result["answer"])

            if result.get("docs"):

                with st.expander("📚 Legal Sources"):

                    for doc in result["docs"]:

                        st.markdown(
                            f"**{doc['meta']['act']} - "
                            f"{doc['meta']['section']}**"
                        )

                        st.text(doc["text"][:800])

            if result.get("trace"):

                with st.expander("🤖 Agent Execution Trace"):

                    for step in result["trace"]:
                        st.write("•", step)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "docs": result.get("docs", []),
            "trace": result.get("trace", [])
        })

    except Exception as e:

        error_message = f"Something went wrong: {e}"

        st.error(error_message)

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message,
            "docs": [],
            "trace": []
        })
