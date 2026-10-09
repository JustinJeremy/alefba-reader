TEAM HANDWRITING - TRAINING DATA (used by train.py).

One photo per letter, cut from each member's clear alphabet sheets
(real_photos/<Member>/<member>_sheet_top.jpg and _bottom.jpg):
  member_handwriting/<Member>/<number>_<name>_top.jpg      from the top half of the sheet
  member_handwriting/<Member>/<number>_<name>_bottom.jpg   from the bottom half

The number at the start is the letter (same numbers as real_photos/README.txt).
train.py shows each photo 15 times per training round (MEMBER_REPEAT), with the usual random changes.

The members' EARLIER sheets (real_photos/<Member>/*_sheet.jpg) are kept as the TEST for these
writers: never copy those here, or the test no longer tells you anything.
To add a new member: make member_handwriting/<Name>/ and put their letters in it with the same
file names, then retrain (python train.py).
