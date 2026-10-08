# Persian Letter Recognition (CNN)

Setup:        pip install -r requirements.txt
Data:         1) python convert_dataset.py <folder with Random1..Random10>   (see below), or Convert IFHCDB letter images to PNGs in data/handwritten/<0..31>/ (class order = LETTERS in common.py; skip digits)
              2) python prepare_data.py   renders the letters in the fonts in fonts/ plus the Arabic-script
                 fonts that come with Windows (read in place, not copied), as photo-like printed samples
Train:        python train.py            (creates persian_cnn.pt, results/metrics.txt, results/confusion_matrix.csv)
                 The Calibri font family is held out of training to measure accuracy on an unseen font.
CLI:          python predict.py img1.jpg img2.jpg img3.jpg   [--ascii = no diacritics]
Frontend:     python app.py   then open http://127.0.0.1:5000
              python app.py --host 0.0.0.0   to open it from a phone on the same Wi-Fi (http://<laptop-ip>:5000)
Output:       romanisation follows the instructor alphabet chart (slide 3), e.g. q for qaf, w for vav
Tests:        python -m pytest              (unit + app + CLI tests, about 30 s)
Score a set:  python score_testset.py rehearsal_set     (or hard_set; test-only images, never train on them)
Real photos:  real_photos/ (TEST ONLY): random_test_photos/ from the app's "Save as test photo", plus JJ/ Ethan/ Mark/ Kaye/
              name files 01_be_anything.jpg (see real_photos/README.txt), then: python score_testset.py real_photos
