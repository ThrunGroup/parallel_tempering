# parallel_tempering

Firstly, install the dependencies using `pip install -r requirements.txt`

Using parallel tempering may improve the overall loss for k-medoids. We're comparing the usage of parallel tempering and the Pypi k-medoids package that outputs the overall loss directly.

For the data, it's currently randomly generated and this data generation is also a part of the `parallel_tempering.py` file.

To run the program on command line with appropriate parameters from argparse, use the following format:

`parallel_tempering.py --verbose True --iterations 100` (you can change the number accordingly to output in the parallel tempering plot)
