
![](https://github.com/scwolof/doepy/blob/master/docs/doepy_logo.png?raw=true)

# Python package for design of experiments.

Currently implements:
* State uncertainty
* Control input uncertainty
* Delta constraints on control input
* Mean (latent and observed) state constraints
* Design criteria
* Discrimination criteria
* SLSQP optimisation
* Model parameter uncertainty

To do:
* Separate time points for controls, measurements and constraints
* optimisation of initial state
* prepare for optimisation of measurement time points
* gradients (continuous time)
* one-model option (prepare for nMPC and DOP?)
* documentation
* tests
* merge branches
* Static models
* Function in ProblemInstance to specify what constraint(s) is violated

## Pronounciation
We pronounce the package name as '_dopey_'

## Authors
* **[Simon Olofsson](https://www.doc.ic.ac.uk/~so2015/)** ([scwolof](https://github.com/scwolof)) - Imperial College London
* **[Eduardo Schultz](http://www.avt.rwth-aachen.de/cms/AVT/Die-AVT/Team/AlleMitarbeiter/~myxr/Schultz-Eduardo/?allou=1)** ([eduschultz](https://github.com/eduschultz)) - RWTH Aachen

## License
The GPdode package is released under the MIT License. Please refer to the [LICENSE](https://github.com/scwolof/doepy/blob/master/LICENSE) file for details.

## Acknowledgements
This work has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no.675251, and from Imperial College London's Data Science Institute Seed Fund.
