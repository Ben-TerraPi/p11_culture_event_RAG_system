import streamlit as st
from rag import answer

def main() -> None:
    st.set_page_config(
        page_title="Chatbot Puls-Events",
        page_icon="🪩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🪩 Chatbot Puls-Events")
    st.caption("Posez une question sur les événements culturels à venir.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Votre question...")
    if not question:
        return
    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            response = answer(question)
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )


if __name__ == "__main__":
    main()