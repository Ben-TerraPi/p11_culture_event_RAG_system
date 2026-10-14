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
│   ├── raw/  clean/  vector_store/              # sorties (ignorées par Git)
│   └── evaluation/
│       └── questions.jsonl         # jeu de questions annoté
├── src/
│   ├── api_utils.py                # cadence des appels + réessais sur 429/5xx
│   ├── chat.py                     # chatbot en ligne de commande
│   ├── clean.py                    # nettoyage, fenêtre de dates, contrôle qualité → data/clean/
│   ├── config.py                   # clés, constantes (départements, période, modèles, seuils)
│   ├── embed.py                    # chunking + embeddings
│   ├── evaluate.py                 # évaluation du RAG sur questions.jsonl
│   ├── extract.py                  # téléchargement API OpenAgenda (brut → data/raw/)
│   ├── main.py                     # orchestration du pipeline
│   ├── rag.py                      # récupération filtrée + génération
│   └── vector_store.py             # index FAISS (cosinus) et recherche
├── tests/
│   ├── test_pipeline_units.py      # nettoyage, chunking (sans clé ni base)
│   ├── test_vector_data.py         # conformité de la base (dates, départements)
│   └── test_vector_store.py        # cohérence et recherche de l'index FAISS
├── .env.example
├── pytest.ini
├── README.md
└── requirements.txt
```

## Reconstruction de la base

Depuis la racine du projet :

```bash
python src/main.py
```

Cette commande extrait les événements, crée `data/clean/agendas.parquet` et reconstruit :

```text
data/vector_store/events.faiss
data/vector_store/metadata.parquet
```

La collecte couvre les cinq départements définis dans `src/config.py` (Côtes-d'Armor, Finistère, Ille-et-Vilaine, Morbihan, Loire-Atlantique) et une fenêtre de 365 jours dans le passé et dans le futur : un an d'historique plus les événements à venir. Pour chaque événement, un seul créneau est conservé, toujours situé dans cette fenêtre (le prochain à venir, sinon le plus récent). Le nettoyage s'arrête avec une erreur si une date ou un département est hors périmètre. Les fichiers générés dans `data/` ne doivent pas être versionnés.

### Limites de l'API Mistral (erreurs 429)

Le palier gratuit est limité en requêtes par seconde et en tokens. Les appels sont donc espacés et réessayés avec un délai croissant en cas de 429 ou d'erreur serveur (`src/api_utils.py`). Les embeddings sont envoyés par petits lots (`EMBED_BATCH`) pour rester sous la limite de tokens par requête. Avec ce palier, la reconstruction complète prend quelques minutes.

## Exécution du chatbot

Après reconstruction de l'index, lancer l'interface conversationnelle :

```bash
python src/chat.py                 # événements à venir uniquement
python src/chat.py --include-past  # inclut aussi les événements passés
```

Le chatbot recherche les chunks pertinents dans FAISS, ne garde qu'un extrait par événement au-dessus d'un seuil de pertinence (`MIN_SCORE`), transmet ce contexte au modèle Mistral via LangChain et répond en français. Il doit rester factuel, ne pas inventer d'événement et indiquer lorsqu'aucun résultat fiable n'est disponible. L'historique n'est pas utilisé dans ce POC.

Une utilisation programmatique suit le même principe. Depuis la racine du projet, rendre `src` accessible dans `PYTHONPATH` avant l'import :

```powershell
$env:PYTHONPATH = "src"
python -c "from rag import answer; print(answer('Quels concerts sont proposés à Rennes ?'))"
```

Dans un script placé dans `src/`, l'appel se présente ainsi, avec des filtres optionnels par lieu et par période :

```python
from rag import answer

print(answer("Quels concerts sont proposés à Rennes ?"))
print(answer("Un concert ?", city="Brest", date_from="2026-11-01", date_to="2026-11-30"))
```

## Évaluation

Le fichier `data/evaluation/questions.jsonl` contient les questions du jeu d'évaluation. Chaque ligne comporte `question`, `expected_answer`, `expected_event_ids` et `notes`. Les réponses annotées et les identifiants d'événements doivent être complétés après une reconstruction de l'index, car les événements collectés peuvent changer (identifiants à relever dans `data/clean/agendas.parquet`). Les questions dont `expected_answer` est vide sont ignorées. Prévoir aussi des questions sans réponse possible, pour vérifier que le chatbot refuse de répondre.

```bash
python src/evaluate.py
```

L'évaluation porte sur la pertinence des événements retrouvés (hit@k), ainsi que sur l'exactitude des dates, lieux et URL et l'absence d'informations inventées, via une note de 0 à 2 attribuée par un modèle Mistral qui compare la réponse à `expected_answer`. Les résultats détaillés sont écrits dans `data/evaluation/results.json`. Ils servent aussi à calibrer `MIN_SCORE`.

## Tests

Lancer toute la suite depuis la racine du projet (`pytest.ini` rend `src` importable) :

```bash
python -m pytest -q
```

- `test_pipeline_units.py` s'exécute sans clé ni base : fenêtre d'un an, départements, doublons, créneaux hors fenêtre, chunking.
- `test_vector_data.py` vérifie que le nombre de lignes de `metadata.parquet` correspond au nombre de vecteurs FAISS, que les champs essentiels sont renseignés, que les départements sont autorisés et que les événements appartiennent à la fenêtre temporelle configurée, calculée à partir de la date de construction de la base.
- `test_vector_store.py` vérifie la cohérence de l'index (vecteurs normalisés, recherche d'un vecteur connu, erreurs explicites).

Les tests qui lisent la base sont ignorés avec un message explicite si l'index n'a pas encore été généré.

## Limites connues

- Les filtres par lieu et par période existent dans `rag.retrieve` mais ne sont pas déduits automatiquement de la question en langage naturel.
- L'index est exhaustif (`IndexFlatIP`) : adapté à quelques milliers de chunks ; au-delà, passer à un index approximatif.