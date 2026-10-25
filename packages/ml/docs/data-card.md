# Data card

What the model is trained and tested on, what was verified by running code, and what is still open.
Numbers below come from the loaders in `dentabridge_ml.data`; the date of each check is given.
Datasets are never committed: `make data` downloads them from the publishers and
`data.lock.json` pins what was unpacked.

## Sources

| Source | Licence | Camera | Labels used | Role |
|---|---|---|---|---|
| AKU caries (Zenodo 14827784) | CC BY 4.0 | one phone (Samsung Galaxy A23, per the publisher) | decay boxes `D` / `d` | train / val / test |
| AlphaDent (Zenodo 16582489) | CC BY 4.0 | DSLR + mirror | filling, crown, abrasion, caries 1-6 | train / val / test (loader pending) |
| Mendeley gingivitis (3253gj88rr) | CC BY 4.0 | clinic camera | image-level gingivitis + 3 clinician captions | train / val / test (loader pending) |
| Kaggle Oral Diseases | Unknown | mixed web and hospital | six classes | private local tests only |

## AKU, checked on 2026-10-25

- 6,265 images in `Dataset.zip` (the Zenodo description says 6,313; the 48 missing images are not
  explained by the publisher). 5,505 have a participant id in the file name, 760 are "pilot".
- 1,122 participants, almost all with five views (frontal, left, right, maxillary occlusal,
  mandibular), one protocol each (with or without retractors). The publisher's own split is by
  image (5,050 / 631 / 631), so photos of one child sit on both sides of it. This project
  splits by participant instead.
- The 760 pilot images have no id. They are grouped by capture order (a new participant starts at
  a "frontal" photo or after a 15-minute pause): 156 sequences, 148 with exactly five views.
  14 of those sequences share near-identical photos with participants of the main set and are
  merged with them. In all, 56 pairs of different participant ids share near-identical photos.
  The largest merged group is 137 images (28 ids), 2% of the data, kept in one split.
- Labels are LabelMe rectangles. `D` (6,228 shapes) is decay in a permanent tooth and `d` (554) in
  a primary tooth. `M` / `m` (45 shapes) are not described by the publisher and are ignored.
  42 of 6,827 shapes lie outside their image (annotated in another, rotated frame): the image
  still counts as positive, the box is dropped. 16 positive images end with no usable box.
- An image with no label file is treated as free of decay, the convention of the publisher's own
  detection benchmark. This is an assumption: it is not stated in the record.
- 2,164 images are positive (34.5 %), with 6,686 usable boxes. **Positives depend heavily on the
  view**: 89 % of mandibular, 66 % of maxillary occlusal, 8 % of right lateral, 7 % of left
  lateral and 3 % of frontal photos. A model can score well by guessing the view, so every
  result is reported per view as well.
- Children and young adults (ages 10-24) from schools in one region; one phone model; image
  sizes range from about 730x410 to 3060x1720.

## Splits

Grouped by participant, never by image (`dentabridge_ml.splits`). Manifests are in
`packages/ml/manifests/` with their sha256. Seed 0 on AKU alone: train 5,015 / val 625 /
test 625 images; 1,738 / 221 / 205 positives.

## Still open

- AlphaDent: image count (paper 1,320 from 295 patients; Hugging Face copy 1,455 rows), whether
  caries grades 1-6 mean severity, whether patient ids exist.
- Mendeley: class labels (are there healthy controls?) and the language of the captions.
  Without negatives, specificity for gingivitis is not estimable and the release gate keeps the
  finding "not assessed".
- DINOv2 licence for the embeddings that group records without an id.
