## Genre in Grammar: PCA on Sanskrit Morphological Variation

### Abstract
Sanskrit literary tradition spans over 2,400 years and encompasses five major genre categories - Vedic, Epic, Kavya, Shastra, and Puranic - whose morphological distinctiveness has never been quantitatively established at scale. In this project we perform large-scale computational analysis of genre-conditioned morphological variation using the Digital Corpus of Sanskrit (DCS), comprising 5.69 million tokens across 270 texts. We apply Principal Components Analysis (PCA) to a 30-dimensional feature vector of normalized morphological and syntactic frequencies per text, revealing a dominant nominalization axis (PC1, 30.3% variance) that separates verbal-heavy Vedic texts from nominal-heavy Shastra and Kavya. Supervised classification with class-weight-balanced Logistic Regression achieves 63.2% macro F1 while unsupervised k-means clustering (ARI $= 0.21$) reveals that only Vedic forms a morphologically coherent cluster. Spearman correlations between text period and morphological features ($r = {-}0.66$ for MCI, $p < 0.0001$) confirm a systematic diachronic shift from morphological complexity toward compounding across Sanskrit's attested history.

Link to the Dataset: [Digital Corpus of Sanskrit](https://github.com/cltk/sanskrit_text_dcs/tree/master/corpora)

To run: 
```
python main.py
```
All the plots will be saved in output/ directory
