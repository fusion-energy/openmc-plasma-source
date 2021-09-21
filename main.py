import numpy as np


class Plasma():
    def __init__(
        self,
        major_radius,
        minor_radius,
        elongation,
        triangularity,
        mode,
        ion_density_centre,
        ion_density_peaking_factor,
        ion_density_pedestal,
        ion_density_separatrix,
        ion_temperature_centre,
        ion_temperature_peaking_factor,
        ion_temperature_beta,
        ion_temperature_pedestal,
        ion_temperature_separatrix,
        pedestal_radius,
        **kwargs,
    ) -> None:

        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.elongation = elongation
        self.triangularity = triangularity
        self.ion_density_centre = ion_density_centre
        self.mode = mode
        self.ion_density_peaking_factor = ion_density_peaking_factor
        self.pedestal_radius = pedestal_radius
        self.ion_density_pedestal = ion_density_pedestal
        self.ion_density_separatrix = ion_density_separatrix

        self.ion_temperature_centre = ion_temperature_centre
        self.ion_temperature_peaking_factor = ion_temperature_peaking_factor
        self.ion_temperature_pedestal = ion_temperature_pedestal
        self.ion_temperature_separatrix = ion_temperature_separatrix
        self.ion_temperature_beta = ion_temperature_beta

    def ion_density(self, r):
        if self.mode == "L":
            density = self.ion_density_centre * \
                (1 - (r/self.major_radius)**2)**self.ion_density_peaking_factor
        elif self.mode in ["H", "A"]:
            if 0 < r < self.pedestal_radius:
                prod = self.ion_density_centre - self.ion_density_pedestal
                prod *= (1 - (r/self.pedestal_radius)**2)**self.ion_density_peaking_factor

                density = self.ion_density_pedestal + prod
            else:
                prod = self.ion_density_pedestal - self.ion_density_separatrix
                prod *= (self.major_radius - r)/(self.major_radius - self.pedestal_radius)
                density = self.ion_density_separatrix + prod

        return density

    def ion_temperature(self, r):
        if self.mode == "L":
            temperature = self.ion_temperature_centre * \
                (1 - (r/self.major_radius)**2)**self.ion_temperature_peaking_factor
        elif self.mode in ["H", "A"]:
            if 0 < r < self.pedestal_radius:
                prod = self.ion_temperature_centre - self.ion_temperature_pedestal
                prod *= (1 - (r/self.pedestal_radius)**self.ion_temperature_beta)**self.ion_temperature_peaking_factor

                temperature = self.ion_temperature_pedestal + prod
            else:
                prod = self.ion_temperature_pedestal - self.ion_temperature_separatrix
                prod *= (self.major_radius - r)/(self.major_radius - self.pedestal_radius)
                temperature = self.ion_temperature_separatrix + prod

        return temperature


def strength(ion_density, ion_temperature):
    return ion_density**2*DT_xs(ion_temperature)


def DT_xs(ion_temperature):
    """Sadler–Van Belle formula
    """
    c = [
        2.5663271e-18,
        19.983026,
        2.5077133e-2,
        2.5773408e-3,
        6.1880463e-5,
        6.6024089e-2,
        8.1215505e-3
        ]
    U = 1.0-ion_temperature*(c[2]+ion_temperature*(c[3]-c[4]*ion_temperature))
    U *= 1/(1.0+ion_temperature*(c[5]+c[6]*ion_temperature))
    val = c[0]/(U**(5/6)*ion_temperature**(2/3))
    val *= np.exp(-c[1]*(U/ion_temperature)**(1/3))
    return val


if __name__ == "__main__":
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
        mode="L",
        shafranov_shift=0.44789,
        triangularity=0.270,
        ion_temperature_beta=6
        )
