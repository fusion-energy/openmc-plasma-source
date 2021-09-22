# openmc-plasma-source

This python-based package offers a way of creating a parametric [OpenMC](https://github.com/openmc-dev/openmc) plasma source from plasma parameters.
The OpenMC sources are ring sources which reduces the computational cost and the settings.xml file size.

![image](https://user-images.githubusercontent.com/40028739/134315320-2188e335-666b-4495-aa88-6b1b049b2df0.png)

The equations implemented here are taken from [this paper](https://doi.org/10.1016/j.fusengdes.2012.02.025).

## Installation

To install openmc-plasma-source, simply run:
```
pip install openmc-plasma-source
```


## Basic usage

```python
from openmc_plasma_source import Plasma


# create a plasma source
my_plasma = Plasma(
    elongation=1.557,
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_pedestal=6.09,
    ion_temperature_separatrix=0.1,
    major_radius=9.06,
    minor_radius=2.92258,
    pedestal_radius=0.8 * 2.92258,
    mode="H",
    shafranov_factor=0.44789,
    triangularity=0.270,
    ion_temperature_beta=6
  )

my_plasma.sample_sources()
my_sources = my_plasma.make_openmc_sources()
```

For a more complete example check out the example script.
