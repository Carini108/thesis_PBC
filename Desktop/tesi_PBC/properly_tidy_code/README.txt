This version of the code is organised in blocks, 
so that the main and the plotting part are shrunk down 
to just a few lines.

To get the code cracking, just type the command 
'make run' and then 'python plots.py' 
to show the results.

Just make sure that the parameters set in the C++ code 
match the ones invoked in the Python script for plotting: 
this step is crucial to correctly identify the files.

-----------------------

SOME NOTES ABOUT THE Makefile:

CXX --> we need this because the bog standard compiler of Apple 
    does not natively support OpenMP for parallelisation

CXXFLAGS --> specific options: C++ standard, optimisation, warnings,
    and libraries and parallelisation habilitation on top of that

LDFLAGS --> linker flags