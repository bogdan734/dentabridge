# Data card

What the model is trained and tested on, what was verified by running code, and what is still open.
Numbers below come from the loaders in `dentabridge_ml.data`; the date of each check is given.
Datasets are never committed: `make data` downloads them from the publishers and
`data.lock.json` pins what was unpacked.

## Sources

| Source | Licence | Camera | Labels used | Role |
|---|---|---|---|---|
| AKU caries (Zenodo 14827784) | CC BY 4.0 | one phone (Samsung Galaxy A23, per the publisher) | decay boxes `D` / `d` | train / val / test |
| AlphaDent (Zenodo 16582489) | CC BY 4.0 | Canon 6D Mark II DSLR with an intraoral mirror | abrasion, filling, crown, caries by site class 1-6 | train / val / test |
| Mendeley gingivitis (3253gj88rr) | CC BY 4.0 | clinic camera | one box per tooth: healthy or gingivitis level 1-4, 3 English captions | loaded, not split (see below) |
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

## AlphaDent, checked on 2026-10-25

- 1,455 JPEGs in the zip: 1,237 train, 83 valid and 135 test. The 135 test images have no labels
  (the hidden test of a competition) and are not used. That explains the 1,320 versus 1,455
  difference: **1,320 labelled images of 293 patients** are in use.
- File names carry a real patient id, sex and age (`p002_F_52_001`). The publisher's own train and
  valid sets share no patient. 2 to 18 photos per patient (median 3); 162 women and 131 men,
  ages 8 to 66 (median 34): adults, which complements AKU's children.
- Labels are YOLO segmentation polygons, reduced here to bounding boxes. Classes: abrasion (6,347
  polygons), filling (2,373), crown (589) and caries by **site, not severity**: the paper grades
  caries in six classes (five standard ones plus a sixth proposed by the WHO), e.g. class 1 is pits
  and fissures, 2 the contact surfaces of molars and premolars, 5 the cervical area, 6 incisal
  edges and cusp tips. The loader keeps them as `caries_1` to `caries_6`; nothing treats a higher
  number as worse, so the urgency rule for a severity subtype (R1) stays disabled.
- 32 polygons have no area (a single point: the class was named, nothing was outlined). The image
  keeps the label and the region is dropped.
- **No image is free of all nine classes.** Every photo shows at least one pathology, so a "negative"
  for caries means "something else is present, not caries". Specificity measured here is against
  that harder negative set, and AKU supplies the caries-free photos instead.
- Positive images per class: abrasion 1,138, caries 910, filling 743, crown 219. Photos are 18 MP
  and are resized to at most 1,536 px before any model sees them.
- Splits for AlphaDent use 70 / 10 / 20 so that the test split holds 44 crown positives (the
  release gate needs 30).

## Mendeley gingivitis, checked on 2026-11-02

- 1,096 images (Training 732, Validation 182, Test 182 by the publisher), no empty label file, a
  caption file per split with three English captions per image (UTF-8 with a BOM).
- Each label file holds one box per tooth (`class cx cy w h`) plus one box for each jaw. The class
  meaning was derived by matching the label counts with the counts in the first caption of every
  image: class 0 and 1 are the upper and lower jaw, 2 is a tooth with healthy gums, 3, 4, 5, 6 are
  gingivitis of level 1, 2, 3, 4 (98-99 % exact agreement per class).
- **Every image shows gingivitis.** 1,077 images have a tooth of level 1-3 and the other 19 only
  level 4; no image has only healthy teeth (309 of about 12,000 tooth boxes are healthy, always
  beside inflamed ones). Image-level sensitivity could be measured, specificity cannot.
- The file numbers are consecutive over all three splits: they form 8 runs, the largest holds 511
  of the 1,096 images. No patient id exists, and runs that size cannot be split into groups.
- So the set is **not part of the v1 splits or training**, and gingivitis stays "not assessed".
  The loader (`dentabridge_ml.data.mendeley`) and the lock keep what the data holds on record.

## Splits

Grouped by participant, never by image (`dentabridge_ml.splits`). Manifests are in
`packages/ml/manifests/` with their sha256. Seed 0, AKU + AlphaDent: 7,585 images in 1,510 groups; train 5,939 /
val 757 / test 889. AKU is split 80 / 10 / 10, AlphaDent 70 / 10 / 20.

## Still open

- DINOv2 licence for the embeddings that would group records without an id (no source needs it
  now: AKU and AlphaDent carry ids).
- A gingivitis-free open photo set: none was found, so gingivitis cannot be assessed yet.
