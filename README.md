# Persian Letter Recognition (CNN)

Setup:        pip install -r requirements.txt
Data:         1) python convert_dataset.py <folder with Random1..Random10>   (see below), or Convert IFHCDB letter images to PNGs in data/handwritten/<0..31>/ (class order = LETTERS in common.py; skip digits)
              2) Put Persian .ttf fonts in fonts/ then: python prepare_data.py
Train:        python train.py            (creates persian_cnn.pt)
CLI:          python predict.py img1.jpg img2.jpg img3.jpg   [--ascii = no diacritics]
Frontend:     python app.py   then open http://127.0.0.1:5000
              python app.py --host 0.0.0.0   to open it from a phone on the same Wi-Fi (http://<laptop-ip>:5000)
Output:       romanisation follows the instructor alphabet chart (slide 3), e.g. q for qaf, w for vav
