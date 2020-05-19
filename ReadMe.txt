Requirements:
numpy, pandas, glob, matplotlib, opencv, matplotlib, keras

---------------------------------------------------------------------
How to use?

Run captcha_solver.py using the below line in command prompt

python captcha_solver.py --input {path to the input directory} --output {path to the output directory} --model {path to the model.hdf5} --labels {path to the labels.pickle file}

Above command will return the detected image and display the detected image and save the solved captcha image to output directory. 
----------------------------------------------------------------------
If you want to train the model again you can use the captcha_solver.ipynb
You can also test the model by using the "Test" section of the notebook.
