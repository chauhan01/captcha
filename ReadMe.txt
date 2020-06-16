Requirements:
numpy, pandas, glob, matplotlib, opencv, matplotlib, keras

---------------------------------------------------------------------
How to use?

Run captcha_solver.py using the below line in command prompt

python captcha_solver.py --input {path to the input directory} --save_solved {True/False, if true save the solved images} --model {path to the model.hdf5} --labels {path to the labels.pickle file}

Above command will return the detected image and save the detected images in "detected output" folder. If "--save_solved" is true then solved captcha image will also be saved to the "solved output" folder. 
----------------------------------------------------------------------
Here is the [Colab Notebook implementation](https://colab.research.google.com/drive/1uydSM5lhbi9Kaq9w1_T6n176PifZwUR0?usp=sharing)
Just run all the cells.
