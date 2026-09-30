# CONTEXT

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.


## Prérequis

- Python 3.13+ 
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))
- Clé API OpenAgenda (obtenue sur [Openagenda Documentation API](https://developers.openagenda.com/))
- [Dataset public opendatasoft](https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/?lang=fr&limit=10&offset=0)

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
├── data/                # sorties (gitigné)
│   ├── raw/  clean/  embedded/
├── src/                  
│   ├── config.py            # clés, constantes (départements, période, modèle)
│   ├── extract.py           # téléchargement API OpenAgenda (brut → data/raw/)
│   ├── clean.py             # nettoyage, dédoublonnage → data/clean/
│   ├── embed.py             # vectorisation Mistral → data/embedded/
│   ├── index.py             # chargement dans la base vectorielle
│   ├── main.py              # orchestration du pipeline
│   └── 
├── README.md
└── requirements.txt
```