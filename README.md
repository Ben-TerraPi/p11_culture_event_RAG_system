# Puls-Events - POC RAG

Ce projet collecte des événements culturels OpenAgenda, les nettoie, les vectorise avec `mistral-embed`, puis les indexe dans FAISS. La couche RAG LangChain utilise cet index pour fournir des recommandations avec un modèle de chat Mistral.


## Prérequis

- Python 3.13+ 
- Clé API Mistral [console.mistral.ai](https://console.mistral.ai/))
- Clé API OpenAgenda* [Openagenda Documentation API](https://developers.openagenda.com/))
*avec accès béta [lecture transverse](https://developers.openagenda.com/evenements/lecture/#lecture-transverse)

## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer un environnement virtuel**

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer les clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY="votre_cle_mistral"
OPENAGENDA_KEY="votre_cle_openagenda" 
```

## Structure du projet

```
.
├── data/                           
│   ├── raw/  clean/  embedded/     # sorties (gitigné)
│   └── evaluation/
│       └── questions.jsonl         # jeu de questions
├── src/
│   ├── chat.py
│   ├── clean.py                    # nettoyage, dédoublonnage → data/clean/
│   ├── config.py                   # clés, constantes (départements, période, modèle)
│   ├── embed.py
│   ├── extract.py                  # téléchargement API OpenAgenda (brut → data/raw/)
│   ├── main.py                     # orchestration du pipeline
│   ├── rag.py
│   └── vector_store.py
├── tests/
│   └── test_vector_data.py
├── README.md
└── requirements.txt
```

## Reconstruction de la base

Depuis la racine du projet :

```bash
python src/main.py
```

Cette commande extrait les événements, crée les fichiers Parquet et reconstruit :

```text
data/vector_store/events.faiss
data/vector_store/metadata.parquet
```

La collecte couvre les cinq départements définis dans `src/config.py` et une fenêtre de 365 jours dans le passé et dans le futur. Les fichiers générés dans `data/` ne doivent pas être versionnés.

## Exécution du chatbot

Après ajout de `src/rag.py` et `src/chat.py`, puis reconstruction de l’index, lancer l’interface conversationnelle :

```bash
python src/chat.py
```

Le chatbot recherche les chunks pertinents dans FAISS, transmet leur contexte au modèle Mistral via LangChain et répond en français. Il doit rester factuel, ne pas inventer d’événement et indiquer lorsqu’aucun résultat fiable n’est disponible. L’historique n’est pas utilisé dans ce POC.

Une utilisation programmatique suit le même principe :

```python
from rag import answer

print(answer("Quels concerts sont proposés à Rennes ?"))
```

## Évaluation

Le fichier `data/evaluation/questions.jsonl` contient les questions du jeu d’évaluation. Chaque ligne comporte `question`, `expected_answer`, `expected_event_ids` et `notes`. Les réponses annotées et les identifiants d’événements doivent être complétés après une reconstruction de l’index, car les événements collectés peuvent changer.

L’évaluation porte sur la pertinence des événements retrouvés, l’exactitude des dates, lieux et URL, ainsi que l’absence d’informations inventées.

## Tests des données vectorielles

Installer les dépendances de développement puis lancer :

```bash
python -m pytest tests/test_vector_data.py -q
```

Le test vérifie que le nombre de lignes de `metadata.parquet` correspond au nombre de vecteurs FAISS, que les champs essentiels sont renseignés, que les départements sont autorisés et que les événements appartiennent à la fenêtre temporelle configurée. Il est ignoré avec un message explicite si l’index n’a pas encore été généré.