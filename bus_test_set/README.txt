BUS PROJECT TESTING DATASET - TEST ONLY, never train on these.

Six small image sets collected for the project (Tests and Compilations.zip, October 2026), renamed to
the project's naming: <class number>_<letter name>_<original name>.jpg. One letter per image.
The Arabic sets use the Arabic kaf (ك) and yeh (ي); they are scored as kaf and ye.

  arabic_print            black Arabic letters on white
  arabic_on_disc          black letters on a turquoise disc (the disc's edge is a ring around the letter)
  cartoon_white           children's-chart cartoon letters: fat, coloured, shiny, dots in another colour
  cartoon_background      the same on a light coloured background
  cartoon_with_pictures   the cartoon letter NEXT TO a picture (lion, apple, carrot, ...): the whole card
  cartoon_with_pictures_boxed   the same cards cut to the right part, where the letter is: what the app
                          reads when you drag a box around the letter
  pastel_tight / pastel_tiles   Persian letters in light pastel colours, small images (61-75 px)
  coloured_tiles_tight / coloured_tiles   black Persian letters on coloured tiles

Score them:   python score_testset.py bus_test_set

Notes
- On a card with a picture, each shape of one strong colour is read on its own and the surest letter
  is kept, when the plain reading of the whole card is unsure (batch 13: 17/28, was 4/28). For the rest,
  drag a box around the letter (or crop the photo) and it is read like any other photo
  (cartoon_with_pictures_boxed).
- The old manuscript photo in the screenshots (joined-up handwriting) is not in here: the app reads
  separate letters, not joined words.
