# SocialChoiceTesting
  This software simulates several prominent social-choice functions with numerous randomly-generated samples of different types of voting populations to learn which vote-tallying techniques statistically tend to have the best desired results for democratic processes. This software includes some of my ideas for new and hopefully better social-choice functions.

  The overall goal of this project is to find a social-choice function that better addresses constraints of election manipulability according to Gibbard's Theorem, and constraints from Arrow's Theorem. Another important consideration is the design of election systems that would maintain the stability and feasibility of governments, and maintain democracy in societies with highly polarized politics. Selecting a social-welfare function that optimizes social welfare relative to such constraints is the overall goal.

  This software is a work in progress, and some parts may still not be functional.

  Work left to do includes finishing simulations, plots, and some debugging.

  This code runs with Python 3 only.  It is not compatible with Python 2.\*

  You can run this simulation by navigating to the folder with the file election_random.py, and executing:
  "python3 -O election_random.py"

  If you want to save the text output from the simulation, run:
  "python3 -O election_random.py &>> summary.txt"

The license for using this software is GPL v3 as written in the included file License.txt.

