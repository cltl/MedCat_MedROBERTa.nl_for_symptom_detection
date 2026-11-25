# SNOMED CT symptom annotation (EN/NL) with MedCAT, MedRoBERTa.nl and MedCATtrainer

Three notebooks take you from raw clinical notes to an evaluated model that recognises symptoms (SNOMED CT concepts) and classifies each mention as **Affirmed** or **Negated**:

| Notebook | Input | Output |
|---|---|---|
| [`01_medcat_prepare_annotation.ipynb`](01_medcat_prepare_annotation.ipynb) | Any set of notes (CSV/XLSX) and a SNOMED CT concept list | MedCATtrainer model pack, plus train and test datasets to annotate |
| [`02_medcat_train_medroberta.ipynb`](02_medcat_train_medroberta.ipynb) | Model pack from 01 and MedCATtrainer **train** export(s) | Full model (NER+L + MedRoBERTa.nl Status classifier), plus a pack for the next annotation round |
| [`03_medcat_evaluate.ipynb`](03_medcat_evaluate.ipynb) | Full model from 02 and MedCATtrainer **test** export(s) | Precision/recall/F1 for concepts and Status, predictions, error lists |

```
notes ──► 01 prepare ──► MedCATtrainer ──► train export ──► 02 train ──► full model ──► 03 evaluate ──► deploy
           │  (pack +     (annotate train   test export ─────────────────────────────────►┘
           │   CSVs)       + test projects)
           └◄──────── next round: 02's annotation pack + new notes ◄──────── 02
```

The older all-in-one notebook `medcat_medcattrainer_snomed_en_nl_medroberta.ipynb` is superseded by these three. It fine-tunes the whole MedRoBERTa.nl encoder, and in tests that setup never learned to tell Affirmed from Negated (see [MetaCAT settings](#metacat-settings)).

All files go to `medcat_data/`:

```
medcat_data/
├── notes/my_notes.csv                     your notes (01 writes a demo set here if missing)
├── annotation_sets/<round>_train.csv      upload to MedCATtrainer (01)
├── annotation_sets/<round>_test.csv
├── exports/<round>_train_export.json      download from MedCATtrainer
├── exports/<round>_test_export.json
├── models/snomed_symptoms_en_nl_annotation_<round|version>.zip   CDB + Vocab: upload to MedCATtrainer
├── models/snomed_symptoms_en_nl_medroberta_<version>.zip         full model: evaluate and deploy
└── evaluation/<model>/                    metrics.json, predictions.csv, gold.csv, per_concept.csv (03)
```

## Two kinds of model pack

| Model pack | Contents | Use it for |
|---|---|---|
| `..._annotation_<tag>.zip` | CDB + Vocab | Uploading to **MedCATtrainer**; starting a next round in 01 |
| `..._medroberta_<version>.zip` | CDB + Vocab + **MedRoBERTa.nl MetaCAT** (Status) | Evaluation (03) and deployment |

MedCAT's MetaCAT only supports BERT models. MedRoBERTa.nl is a RoBERTa model, so notebooks 02 and 03 redirect MedCAT's BERT loaders to Hugging Face `AutoModel`/`AutoTokenizer` before they create or load a MetaCAT:

```python
from transformers import AutoModel, AutoTokenizer
import medcat.components.addons.meta_cat.models as _mc_models
import medcat.components.addons.meta_cat.mctokenizers.bert_tokenizer as _mc_bert_tok
_mc_models.BertModel = AutoModel
_mc_bert_tok.BertTokenizerFast = AutoTokenizer
```

MedCATtrainer runs a standard MedCAT install without this patch, so it gets the pack without MetaCAT. Annotators label Status there with a meta task. **Any Python code that loads a `_medroberta_` pack needs the patch above first.**

---

## 1. Set up

### 1.1 Python environment

The notebooks run in `../venv` (MedCAT 2.x):

```bash
../venv/bin/pip install "medcat[meta-cat]" transformers torch accelerate pandas scikit-learn
```

You don't need spaCy, because MedCAT 2 uses its built-in regex tokenizer by default. For XLSX input, also install `openpyxl`. The first run downloads `CLTL/MedRoBERTa.nl` (~500 MB) into the Hugging Face cache.

> **Editing a notebook outside Jupyter:** if a notebook file was changed on disk (for example by Claude) while it was open in Jupyter, reload it from disk **without saving first**. Saving from Jupyter overwrites the file with the old version that Jupyter has in memory.

### 1.2 Start MedCATtrainer

MedCATtrainer lives in `../MedCATtrainer` and uses image `cogstacksystems/medcat-trainer:medcat-trainer-v3.9.0`. You need **v3.x**: the v2.x images run MedCAT 1, which can't read MedCAT 2 model packs and fails with `No such file or directory: .../cdb.dat`.

```bash
cd ../MedCATtrainer
docker compose up -d
```

- The v3 images are only published for amd64. `docker-compose.yml` sets `platform: linux/amd64`, so on Apple Silicon the container runs under emulation and is slower.
- On the first start after an upgrade you may see `duplicate column name` in the `[bg-process]` log. Two processes are migrating the database at the same time. Run `docker compose restart medcattrainer` once; afterwards both log `No migrations to apply`.
- The upstream project has moved to [CogStack/cogstack-nlp](https://github.com/CogStack/cogstack-nlp/tree/main/medcat-trainer).

Open the admin site at **http://localhost:8001/admin/**. The default login is `admin` / `admin`. **Change it** (Users → admin → change password) before loading real patient text. Create a separate, non-staff user for each annotator.

---

## 2. Prepare an annotation round (notebook 01)

Set the configuration cell, then run all cells:

| Setting | Meaning |
|---|---|
| `NOTES_PATH`, `TEXT_COLUMN`, `NAME_COLUMN` | Your notes: CSV or XLSX, one note per row. Use **de-identified** text. Document names must be unique (duplicates get a suffix; missing names are generated). |
| `TEST_FRACTION` | Share of notes held out as the test set for notebook 03 (default 0.2). |
| `CONCEPTS_CSV` | MedCAT concept CSV with the columns `cui,name,ontologies,name_status,type_ids` (`P` = preferred name, `A` = synonym). `None` uses a starter list of 5 symptoms. Replace it with a SNOMED CT export once the Nictiz licence is in place. |
| `START_FROM_MODEL_PACK` | For later rounds: the `..._annotation_<version>.zip` from notebook 02. |
| `ROUND_TAG` | Names the output files (`round1`, `round2`, ...). |
| `OVERWRITE_EXISTING` | `False` (default) stops the notebook instead of replacing a pack that already exists under that tag. |

The notebook:

1. builds the concept database (CDB),
2. gives every word in the notes a vector from MedRoBERTa.nl (MedCAT uses word vectors to tell concepts apart from context),
3. runs unsupervised training on the train notes only, so the test notes stay unseen,
4. saves `models/..._annotation_<round>.zip` and `annotation_sets/<round>_train.csv` / `_test.csv`,
5. shows the spans MedCATtrainer will suggest for a few notes.

### Configure MedCATtrainer for the round

In the admin site:

1. **Model packs → Add model pack:** upload `medcat_data/models/..._annotation_<round>.zip`. Don't upload a `_medroberta_` pack: its MetaCAT fails to load.
2. **Meta tasks → Add meta task** (once): name it `Status`, add the values `Affirmed` and `Negated` (with the `+` icon), and choose a default. These names must match notebook 02 (`STATUS_TASK`, `STATUS_VALUES`).
3. **Datasets → Add dataset:** upload `<round>_train.csv`, and separately `<round>_test.csv`.
4. **Project annotate entities → Add:** create **two projects**, both with the model pack, the `Status` task (in *Tasks*, Ctrl/Cmd-click) and the annotator users:

   | | Train project | Test project |
   |---|---|---|
   | Dataset | `<round>_train.csv` | `<round>_test.csv` |
   | Train Model On Submit | optional (better suggestions while annotating) | **off**, so the test set doesn't feed into the model |
   | Require Entity Validation | on | on |
   | Cuis (optional) | e.g. `267036007,29857009,386661006,84229001,49727002` | same |

   Notebook 03 ignores predicted concepts outside a project's *Cuis* filter, since annotators never saw them.

---

## 3. Annotate

Annotators log in at **http://localhost:8001/** and open the project. For each document:

1. **Review every highlighted span.** Grey spans are MedCAT suggestions that nobody has reviewed yet. Move through them with ←/→ or by clicking, and give each one a status:
   - **Correct:** the span and its SNOMED CT concept are right (shown in blue).
   - **Incorrect:** the span isn't this concept (red).
   - **Terminate:** this text should *never* link to this concept (dark red).
   - **Alternative:** right span, wrong concept. Pick the correct concept with the concept picker (turquoise).
   - **Irrelevant** (if enabled): the mention doesn't matter for this task.
2. **Set Status** on every *correct* span: **Affirmed** if the patient has the symptom (*"klaagt over kortademigheid"*), **Negated** if it's ruled out (*"geen koorts"*, *"denies chest pain"*). The meta task panel only appears for spans marked correct.
3. **Add missed mentions:** select the text, right-click → **Add Term**, search for the concept by name or SNOMED ID, and add it. Then mark it correct and set its Status. This matters most for synonyms, abbreviations and spelling variants that aren't in the CDB yet.
4. **Submit** when all spans are reviewed (Submit stays disabled until then). The Reset button clears a document if it went wrong.

Guidelines for consistent labels:

- Annotate the shortest span that names the symptom (`pijn op de borst`, not `ernstige pijn op de borst`).
- Negation cues include *geen, ontkent, zonder, niet, no, denies, without*. Mark spans under these cues **Negated** even if the cue is a few words away (*"geen koorts of hoesten"*: both are negated).
- Annotate the test project as carefully as the train project: its labels are what the model is scored against.
- MetaCAT needs examples of both classes. Aim for at least a few hundred Status labels, with a reasonable number of Negated ones.

### Export

In the admin site, go to **Project annotate entities**, tick **one** project, choose **Download annotations** (the *with text* option) from the action menu, and press **Go**. Save the files as:

- `medcat_data/exports/<round>_train_export.json` for notebook 02
- `medcat_data/exports/<round>_test_export.json` for notebook 03

The exports contain the full text of every document, so store them like the source notes.

---

## 4. Train (notebook 02)

Set `BASE_MODEL_PACK` (the `..._annotation_<round>.zip` you annotated with) and `TRAIN_EXPORTS`, then run all cells. The notebook:

1. checks the export (documents, correct annotations, Status label counts) and stops if Status values don't match `STATUS_VALUES`,
2. runs **supervised NER+L training** on the validated spans (`NER_EPOCHS`),
3. trains the **MetaCAT Status classifier** on MedRoBERTa.nl and prints a validation report (from `METACAT_VAL_FRACTION` of the train data),
4. saves `models/..._medroberta_<VERSION_TAG>.zip` (full model) and `models/..._annotation_<VERSION_TAG>.zip` (for the next round),
5. shows predictions for two example sentences.

The MetaCAT is trained from scratch on every run. For a new round, list **all** train exports so far in `TRAIN_EXPORTS`, and give the run a new `VERSION_TAG`. **Never list a test export here.**

### MetaCAT settings

MetaCAT runs MedRoBERTa.nl over the mention and up to `CONTEXT_TOKENS` (15) tokens of context on each side, and trains a classifier on the result.

| Setting | Default | Notes |
|---|---|---|
| `FREEZE_ENCODER` | `True` | Keep MedRoBERTa.nl fixed and train only the classifier head |
| `METACAT_LR` | `1e-3` | Learning rate for the classifier head |
| `METACAT_EPOCHS` | 10 | The best epoch on the validation split is kept |
| `METACAT_BATCH_SIZE` | 8 | |
| class weights | on | Compensates when Negated is rarer than Affirmed |

These defaults come from tests on synthetic Dutch/English annotations (160 Status labels, 56 test labels):

| Setup | Test F1 (Status) |
|---|---|
| Encoder frozen, lr 1e-3, 10 epochs | **0.93–0.95** |
| Whole encoder fine-tuned, lr 2e-5 / 3e-5 / 1e-4, 10–30 epochs | 0.25–0.35: predicts one class for everything |

Full fine-tuning may work with much more data, but check its validation report: if one class gets recall 0, it isn't learning.

---

## 5. Evaluate (notebook 03)

Set `MODEL_PACK` (the `..._medroberta_<version>.zip`) and `TEST_EXPORTS`, then run all cells. It reports:

1. **Concept recognition and linking (NER+L):** precision, recall and F1, overall and per concept.
   - *Strict*: the predicted span and concept exactly match an annotation.
   - *Lenient*: the spans overlap and the concept is the same.
   - Gold annotations are the validated ones marked correct or alternative. Spans annotators marked incorrect, terminated or irrelevant count as false positives if the model still predicts them.
2. **Status end to end:** for gold mentions the model found, a classification report and confusion matrix of its Affirmed/Negated labels.
3. **Status on the annotated spans:** the MetaCAT alone (`MetaCAT.eval`) on every annotated mention. This separates classifier errors from NER errors.

The results go to `medcat_data/evaluation/<model>/`:

- `metrics.json`: all metrics, with the model and export names
- `predictions.csv` / `gold.csv`: every mention, with its match type (`strict`, `lenient`, `FP`, `FN`) and Status
- `per_concept.csv`

The notebook ends with lists of missed mentions, spurious mentions and wrong Status labels for error analysis.

### Next round

1. Collect new notes.
2. In notebook 01, set `START_FROM_MODEL_PACK` to `models/..._annotation_<version>.zip` from notebook 02, and give the round a new `ROUND_TAG`.
3. Annotate and export as before.
4. Train with **all** train exports and a new `VERSION_TAG`.
5. Evaluate on the new test export. To compare models, also evaluate on the earlier test exports.

---

## 6. Deploy a trained model

The deliverable is the **full model pack** (`..._medroberta_<version>.zip`). It contains the CDB, the vocabulary, the config and the MedRoBERTa.nl MetaCAT, including its weights and tokenizer. Deploy the version that scored best in notebook 03.

### 6.1 Requirements on the target machine

- Python with `medcat[meta-cat]` **2.x**, the same minor version used for training (2.10 here), plus `transformers` and `torch`.
- Internet access to Hugging Face on first load, or a pre-filled HF cache (`HF_HOME`). When it rebuilds the model, MetaCAT fetches the `CLTL/MedRoBERTa.nl` config and base weights, then overwrites the weights with the trained ones from the pack. For offline servers, copy the cache and set `HF_HUB_OFFLINE=1`.
- The MedRoBERTa.nl patch, applied **before** `CAT.load_model_pack`.

### 6.2 Use it from Python

```python
from transformers import AutoModel, AutoTokenizer
import medcat.components.addons.meta_cat.models as _mc_models
import medcat.components.addons.meta_cat.mctokenizers.bert_tokenizer as _mc_bert_tok
_mc_models.BertModel = AutoModel                 # MedRoBERTa.nl is RoBERTa, not BERT
_mc_bert_tok.BertTokenizerFast = AutoTokenizer

from medcat.cat import CAT

cat = CAT.load_model_pack("models/snomed_symptoms_en_nl_medroberta_v1.zip")


def annotate(text: str) -> list[dict]:
    ents = cat.get_entities(text)["entities"].values()
    return [{
        "start": e["start"], "end": e["end"], "text": e["source_value"],
        "snomed_id": e["cui"], "name": e["pretty_name"], "acc": e["acc"],
        "status": e["meta_anns"]["Status"]["value"],
        "status_confidence": e["meta_anns"]["Status"]["confidence"],
    } for e in ents]


print(annotate("Patiënt heeft koorts, maar geen pijn op de borst."))
```

For many documents, use `cat.get_entities_multi_texts(texts)`. It returns a generator of `(index, result)` pairs, where `index` is the text's position as a string and `result["entities"]` has the same structure as above:

```python
for idx, result in cat.get_entities_multi_texts(texts):
    print(texts[int(idx)][:40], [e["cui"] for e in result["entities"].values()])
```

### 6.3 As an HTTP service

Wrap `annotate` in a small web service, so other systems can send notes without needing Python or MedCAT. Install the extra packages:

```bash
pip install fastapi uvicorn
```

Then create `serve.py` (it includes the patch, load and `annotate` code from 6.2):

```python
from fastapi import FastAPI
from pydantic import BaseModel
# ... patch + CAT.load_model_pack(...) + annotate() from 6.2 ...

app = FastAPI()


class Note(BaseModel):
    text: str


@app.post("/annotate")
def annotate_endpoint(note: Note) -> list[dict]:
    return annotate(note.text)
```

Run the service:

```bash
uvicorn serve:app --host 127.0.0.1 --port 8090
```

Then call it:

```bash
curl -s localhost:8090/annotate -H 'Content-Type: application/json' -d '{"text": "Geen koorts, wel hoesten."}'
```

- Load the model once at startup (as above), not once per request.
- Run a single worker per GPU or CPU budget. Each worker holds its own copy of MedRoBERTa.nl in memory.
- Keep the service on `127.0.0.1` or behind an authenticated proxy inside the hospital network. Clinical text must not leave it.
- MetaCAT runs on the CPU by default. To use a GPU, override it at load time with `CAT.load_model_pack(path, addon_config_dict={"meta_cat.Status": {"general": {"device": "cuda"}}})`.
- CogStack's ready-made MedCAT services use a stock MedCAT install without the patch, so they can't load the MedRoBERTa.nl MetaCAT. They can serve the `_annotation_` pack, which does concept recognition only, without Status.

### 6.4 Before using it for real

- Base the deployment decision on notebook 03's results on a test set annotated by clinicians, not on notebook 02's validation report.
- Record which model pack, MedCAT version and annotation exports each deployment uses. The pack's `model_card.json` records the MedCAT version and training history, and notebook 03 writes `metrics.json`.
- The starter CDB only holds 5 symptom concepts. Once the Nictiz SNOMED CT licence is in place, set `CONCEPTS_CSV` in notebook 01 to the full Dutch/English concept list and start a new round.
