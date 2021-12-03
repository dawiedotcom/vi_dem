# VI DEM

This is the companion code to 
[A Variational Integrator for the Discrete Element Method](https://arxiv.org/abs/2103.01757).

Figures for the various examples in the paper can be generated
by running the scripts in `examples`. 
Comparisons to LAMMPS can be made by running the `run.sh`
scripts in subdirectories of the `lammps` folder, provided
that LAMMPS is installed and available through the `lmp` command.
The figures can be plotted using LaTeX and pgfplots by running
`make` in the top level directory.

## Environment

The Python environment is managed through `virtualenv`. The following
command line steps will download the source code, create the 
python environment and run an example.

    # Clone the git repo
    git clone https://github.com/dawiedotcom/vi_dem
    cd qc_dem

    # Create a fresh Python 3 environment
    virtualenv -p python3 venv
    source venv/bin/activate

    # Install Python packages
    pip install -r requirements.txt

    # Run some examples
    python examples/impact_analytic.py


## License

This software is distributed under the MIT License. See LICENSE for details.

    
