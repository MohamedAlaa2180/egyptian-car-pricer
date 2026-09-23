# Egyptian car pricer

Fine-tune a small open-source LLM to estimate an **Egyptian-market asking price in EGP** from a short spec sheet (brand, model, year, mileage, fuel, transmission).

This follows the same recipe as Ed Donner's Week 7 **The Price is Right** project: prompt/completion SFT, QLoRA, then eval with absolute error. It is **not** a universal car model, **not** a formal valuation, and **not** an Egyptian-law assistant.

**Disclaimer:** outputs are listing-style estimates from noisy classified ads. They are not appraisals, insurance values, or legal advice.

## Dataset

Source (already prompt/completion formatted): [mo-hug-me/Egyptian_cars_price_prediction](https://huggingface.co/datasets/mo-hug-me/Egyptian_cars_price_prediction)

The Hub file has its own train (14,734), validation (4,999), and test (4,999) splits. The Hub test completion is `0` on every row, so those ads have no price label and are not used. We pool Hub train and validation, apply the same filters, then hold out **1,000 validation** and **1,000 test** rows. Training uses the rest.

Inference prompt ends with `Price is EGP`. The model should continue with an integer EGP amount. Do not include the number in the prompt at eval time.

## Open this as a Cursor workspace

This is a **separate repo** from `llm_engineering`. In Cursor: **File → Open Folder** → `e:\AI_Projects\egyptian-car-pricer`.

Keep the course repo open in another window if you want to compare `week7/day2.ipynb` and `week7/pricer/`.

## Setup (Windows, data prep)

From this folder, using the same Python toolchain you use for the course (venv or uv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
```

Put `HF_TOKEN` in `.env` if the Hub rate-limits anonymous downloads. Put `WANDB_API_KEY` in `.env` for the training run. **Do not commit `.env`.**

## How to run (milestone order)

1. **Data prep (now)** — `notebooks/01_inspect_and_prep.ipynb`  
   Load the Hub dataset → parse fields → apply filters → train/val/test split → save under `data/` (gitignored).  
   Do **not** push to the Hub unless you explicitly ask to.
2. **Train** — `notebooks/02_train_qlora_colab.ipynb` on a Google Colab T4.  
   Model: `Qwen/Qwen2.5-3B` (base, not Instruct), QLoRA 4-bit NF4 + LoRA.  
   First run is one epoch on the 8,000-row lite set. Loss curves go to the Weights & Biases project `mohamedalaasalem1/egyptian-car-pricer`. The 1,000 test cars stay out of this notebook.
3. **Eval** — `notebooks/03_eval.ipynb` on a Colab T4.  
   1,000-row held-out test, MAE + MAPE vs the lite-train mean, the lite-train median, unadapted Qwen, and the QLoRA adapter. The lite run's best checkpoint was step 500.

`LITE_MODE` uses 8,000 train rows so a T4 run is cheap. Full mode uses all remaining train rows. The **same 1,000 test cars** are kept in both modes.

## Layout

```
pricer/                 # Week 7-style helpers (EGP prefix, Tester, cleaning)
notebooks/              # inspect → train → eval
data/                   # local parquet only (gitignored)
```

## What we will not do

- Scrape Hatla2ee / OLX
- Convert prices to USD
- Commit secrets, adapters, or model weights
- Push datasets or models to the Hub without asking
