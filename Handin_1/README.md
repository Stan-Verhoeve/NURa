# Solutions to hand-in 1
These are my solutions to hand-in 1 for the NURa course. I have organized my code as follows:
- Main scripts have the corresponding question as leading number in their name (e.g. [01\_poisson\_distribution.py](01_poisson_distribution.py) for question 1)
    - In these main scripts, various functions and classes for that script are defined
    - These main scripts run all subquestions without distinction / indication. For distinction, see the PDF file generated through LaTeX.
- Helper routines can be found in the folder [helper\_scripts](helper_scripts). These include
    - Neville interpolation scheme ([interpolation.py](helper_scripts/interpolation.py))
    - Pretty printing functions ([pretty\_printing.py](helper_scripts/pretty_printing.py))


### Usage
The `run.sh` file runs all python scripts, prints their output to terminal, and builds the `.pdf` file which has additional explanation for each (sub)question.
