# Dental datasets and benchmarks for a patient-photo model
Checked 2026-10-25. P = page or paper read, S = search summary only, UNVERIFIED = not confirmed. Not legal advice.
Method: one researcher pass (8 searches, 6 fetches) plus an owner-session re-check of MetaDent (3 fetches).

## 1) Question and constraints
Which public datasets can train or evaluate a model that reads a patient's own photo of teeth or mouth (smartphone and intraoral colour photos first; panoramic, bitewing and periapical X-ray second) and reports findings such as caries, calculus, gingivitis, discolouration, missing teeth, ulcers, restorations, malocclusion? Also: dental VQA sets and VLM benchmarks. Hard filters: free to download, licence compatible with publishing trained weights, no real patient data collected by us.

## 2) Options
| Option | What it is | Fit | Cost and risk | Source |
|---|---|---|---|---|
| Zenodo / AKU caries set | 6,313 JPG intraoral photos, boxes (YOLO, COCO, VOC, LabelMe) in two classes: `d` primary-tooth decay, `D` permanent-tooth decay; ages 10-24, government schools in Mithi, Sindh (Pakistan), one Samsung Galaxy A23; split 5,050/631/631; two dentists (>2 years) plus a third for disagreements, kappa 0.89 on 10%; ethics approval AKU ERC#2024-9433-29648 with consent for open sharing. Zenodo record: CC BY 4.0 (the Sci Data article itself is CC BY-NC-ND, cite the record). 3.1 GB, published 2025-02-07 | Best photo set: phone camera, permissive licence, large | Decay only, no severity. Patient count not stated (about 630 if ten photos per patient, inference). The 80/10/10 sizes match a split by image, so photos of one mouth probably cross train and test: leakage; we must regroup before splitting. Single site, young patients, resolution not stated | [4] P |
| AlphaDent | 1,320 DSLR and mirror intraoral photos of 295 patients (the HF release has 1,455 rows, difference unexplained); instance masks; 9 classes: abrasion, filling, crown, caries 1-6. Licence verified on two primary pages: Zenodo 10.5281/zenodo.16582489 CC BY 4.0 (4.9 GB), HF ZFTurbo/AlphaDent Apache-2.0. Code Apache-2.0 at github.com/ZFTurbo/AlphaDent with pretrained YOLOv8x weights | Restorations, wear and caries stages that AKU lacks; ready baseline weights to compare against | Splits and patient-level grouping not stated on Zenodo or HF; the Kaggle competition keeps test labels hidden; clinic camera, small | [1] P, [13] P |
| Mendeley gingivitis set | 1,096 high-resolution intraoral images of gingivitis, each with a label and three captions by periodontists; train/val/test structure; Hanoi Medical University; 6.1 GB; CC BY 4.0 (page read); published 2024-07-31 | Only verified open set with clinician text and gum findings | Gingivitis only, no healthy controls stated; captions may be in a language other than English (not checked) | [14b] P |
| MetaDent | 60,669 intraoral images, 2,588 annotated (overall summary + free-text labels + segmentation), about 15k-18k VQA pairs, 18-class multilabel; JDR 2026. Images from clinical, public and web sources. Full set on Hugging Face and ModelScope; benchmark code at github.com/WenhDeng/metadent-bench | Closest match to our VLM task (photo VQA, captions) | Dataset licence not stated on the project page; web-sourced images may carry mixed rights; camera type and split not stated. Treat as eval-only until the HF dataset card is read | [12] P |
| Kaggle "Oral Diseases" | Page read: six classes (caries, calculus, gingivitis, discolouration, ulcers, hypodontia) with boxes; images "from multiple hospitals and reputable dental websites", augmentation (rotation, flip, scale, noise) already baked in | Only set with calculus, ulcers, discolouration, hypodontia | **Licence: Unknown** on the Data Card, web-sourced and pre-augmented (copies of one photo can sit in train and test). Excluded from training and from published weights | [5] P |
| Roboflow Universe dental projects | Per-project calculus, caries, gingivitis sets | Quick extras | Per-project licence and provenance unstated | [7] S |
| IO150K | More than 150k intraoral photos for tooth segmentation | Tooth localisation | Licence UNVERIFIED | [8] S |
| DENTEX | 1,005 labelled panoramic X-rays plus partially labelled ones; caries, deep caries, periapical lesion, impacted tooth, FDI numbering; CC BY-NC-SA 4.0, open on HF | Only verified X-ray set | NC and share-alike: weights trained on it need an NC model card (conservative reading) | [10] P |
| Tufts Dental Database | 1,000 panoramics, expert labels, eye-tracking | X-ray extra | Licence and access route UNVERIFIED | [11] P |
| MMOral-OPG-Bench | 100 panoramics, 1,069 questions, MIT tag, eval only | X-ray VQA eval | Image provenance unstated, so MIT does not clear upstream rights | [12b] P |
| OralGPT-Omni, DentalGPT | Fine-tuned dental VLMs and their data | Prior art to beat | Data and weights "will be released", licences UNVERIFIED | [14][15] |
| MouthCare HF sample | Intraoral sample | none | Gated, bars model training | [9] S |

Reference points for the README (author-reported or small n, context only):
- Gemini 3.1 on 1,255 public intraoral photos: image-level F1 0.87, tooth-level F1 0.63 (S, Int Dent J 2026) [17].
- OralGPT-Omni table, intraoral column, authors' LLM-judged benchmark: GPT-5 44.6, Claude Sonnet 4.5 36.3, Qwen3-VL-8B 14.3, Qwen2.5-VL-7B 11.2, their fine-tuned model 66.8 (P, tool-parsed, re-verify before quoting) [14].
- GPT-4o on 50 panoramics: F1 0 for caries, periapical lesions, fractures (S) [18].
- Independent store-and-forward photo study: sensitivity 57%, specificity 100%, n=6 (S).

## 3) Recommendation: reuse open data, build labels and evaluation ourselves
Train on three licence-verified photo sets: Zenodo/AKU (phone photos, decay boxes), AlphaDent (restorations, wear, caries stages, masks) and the Mendeley gingivitis set (gum findings with clinician captions), all CC BY 4.0 or Apache-2.0, so the weights can stay open. Evaluate on a held-out split we build ourselves, grouped by pseudo-patient (embedding clusters), because the AKU split is by image. v1 findings: caries (decay, stages where available), filling, crown, abrasion, gingivitis. Calculus, ulcers, discolouration, hypodontia and malocclusion have no licence-clean open data, so the model reports them as "not assessed". Urgency hints come from deterministic rules, not from the model.
X-ray stays optional: DENTEX forces an NC + share-alike model card.
No verified open VQA training set exists: build templated VQA from box and mask labels and keep a per-dataset licence table.

## 4) Improvements worth trying
- Patient-level splits, duplicate and near-duplicate check (perceptual hash) before any training; a test per rule (the owner's own habit: every bug gets a test that fails without the fix).
- Report per-class sensitivity and specificity with confidence intervals, plus a phone-versus-clinic-camera shift check (AKU vs AlphaDent).
- Read the MetaDent HF dataset card; if licence and provenance are clean, use it for VQA and captions.

## 5) Open questions for the owner
1. Must the published weights allow commercial use? If yes, DENTEX and any NC set are out (the three photo sets above are fine).
2. Accept a v1 whose findings are limited to caries, restorations, wear and gingivitis?
3. Other leads, licence not checked (S): malocclusion set (4,166 photos, 384 participants, arXiv 2505.15637), IOP-Compass (5,000 photos, tooth masks only), DentalAI (2,495 images, tooth/caries/cavity/crack).

## 6) Sources (checked 2026-10-25)
[1] https://arxiv.org/abs/2507.22512 AlphaDent, 2025-07, S | [2] https://arxiv.org/abs/2603.28297 DinoDental, 2026-03, S | [3] https://arxiv.org/abs/2604.23718 2026-04, S
[4] https://zenodo.org/records/14827784 Zenodo/AKU, 2025-02-07, P; paper doi 10.1038/s41597-025-05647-9, S
[5] https://www.kaggle.com/datasets/salmansajid05/oral-diseases P (Data Card read 2026-10-25) | [13] https://zenodo.org/records/16582489 and https://huggingface.co/datasets/ZFTurbo/AlphaDent and https://github.com/ZFTurbo/AlphaDent P | [14b] https://data.mendeley.com/datasets/3253gj88rr/1 P | [7] https://universe.roboflow.com/skripsi-fix/oraldiseasesagain S | [8] https://arxiv.org/abs/2404.01013 S | [9] https://huggingface.co/datasets/MouthCare/intraoral-sample S
[10] https://huggingface.co/datasets/ibrahimhamamci/DENTEX P (paper arXiv 2305.19112, 2023) | [11] https://tdd.ece.tufts.edu/ P, undated
[12] https://arxiv.org/abs/2604.14866 and https://menxli.github.io/metadent/ and https://github.com/WenhDeng/metadent-bench MetaDent, 2026-04, P | [12b] https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench P
[14] https://arxiv.org/html/2511.22055v1 OralGPT-Omni, 2025-11, P | [15] https://arxiv.org/abs/2512.11558 DentalGPT, 2025-12, S
[17] https://escholarship.org/uc/item/1z1973zj Moharrami 2026, S | [18] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12468776/ 2025, S
