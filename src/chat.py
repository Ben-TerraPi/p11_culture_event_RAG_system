from rag import answer

print("Chatbot Puls-Events. Tapez 'quit' pour quitter.")

while True:
    question = input("\nVous : ").strip()

    if question.lower() in {"quit", "exit"}:
        break

    print(f"\nAssistant : {answer(question)}")