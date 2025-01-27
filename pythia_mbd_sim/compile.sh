#!/bin/bash

# Compile the program
g++ -o main main.cpp $(pythia8-config --cxxflags --ldflags) $(root-config --cflags --libs)

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable created: ./main"
else
    echo "Compilation failed."
fi

