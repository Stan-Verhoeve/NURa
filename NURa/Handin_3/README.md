# Solutions to hand-in 3
These are my solutions to hand-in 3 for the NURa course. I have organized my code as follows:
- Main scripts have the corresponding question as leading number in their name
    - In these main scripts, various functions and classes for that script are defined
    - These main scripts run all subquestions without distinction / indication. For distinction, see the PDF file generated through LaTeX.
- Helper routines can be found in the folder [helperscripts](helperscripts). These include
    - Pretty printing functions ([prettyprint.py](helperscripts/prettyprint.py))
    - A lot more. Too long to include here


### Usage
The `run.sh` file runs all python scripts, prints their output to `.txt` files (saved in `OUT/`), and builds the `.pdf` file which has additional explanation for each (sub)question. The `.pdf` file can be found in [latex/OUT](latex/OUT). Note that [latex/OUT](latex/OUT) will be created DURING executing `run.sh`.
