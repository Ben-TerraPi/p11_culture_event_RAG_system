from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

import vector_store
from config import FAISS_FILE, METADATA_FILE


def retrieve(query: str, limit: int = 5) -> list[Document]:
    results = vector_store.search(
        query=query,
        index_file=FAISS_FILE,
        metadata_file=METADATA_FILE,
        limit=limit,
    )

    documents = []
    for _, row in results.iterrows():
        content = (
            f"Titre : {row.get('title')}\n"
            f"Date : {row.get('date_start')}\n"
            f"Ville : {row.get('city')}\n"
            f"Lieu : {row.get('venue')}\n"
            f"Description : {row.get('description')}\n"
            f"URL : {row.get('url')}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "event_id": row.get("event_id"),
                    "score": row.get("score"),
                    "url": row.get("url"),
                },
            )
        )

    return documents


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Tu es l'assistant événementiel de Puls-Events.
Réponds en français à partir exclusivement du contexte fourni.
Ne crée jamais d'événement, de date, de lieu ou d'URL absent du contexte.
Si aucune information ne permet de répondre, indique-le clairement.
Présente les recommandations de façon concise et précise.

Contexte :
{context}
""",
    ),
    ("human", "{question}"),
])

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
)

chain = prompt | llm | StrOutputParser()


def answer(question: str) -> str:
    documents = retrieve(question)
    context = "\n\n".join(document.page_content for document in documents)

    if not context:
        return "Je n'ai trouvé aucun événement correspondant à votre demande."

    return chain.invoke({
        "question": question,
        "context": context,
    })