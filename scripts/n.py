from tempfile import NamedTemporaryFile
import yaml
import numpy as np
from astropy import constants

from photochem.clima import AdiabatClimate
from photochem.utils import stars
from photochem.utils import settings_dict_for_climate, species_dict_for_climate

# conda create -n test -c conda-forge photochem=0.8.1

class AdiabatClimateSimple(AdiabatClimate):

    def __init__(
        self, 
        *,
        M_planet=1.0, 
        R_planet=1.0, 
        Teff=5700.0,
        nz=50, 
        number_of_zeniths=4
    ) -> None:
        
        species_dict = species_dict_for_climate(
            species=['CO2','N2'],
            condensates=[]
        )

        opacities = {
            'k-distributions': True, 
            'CIA': True, 
            'rayleigh': True, 
            'photolysis-xs': True
        }
        settings_dict = settings_dict_for_climate(
            planet_mass=float(M_planet*constants.M_earth.to('g').value), 
            planet_radius=float(R_planet*constants.R_earth.to('cm').value), 
            surface_albedo=0.0, 
            number_of_layers=int(nz), 
            number_of_zenith_angles=int(number_of_zeniths), 
            photon_scale_factor=1.0, 
            opacities=opacities
        )

        # Prepare stellar spectrum
        stellar_flux = 1360.0
        wv_planet, F_planet = blackbody_spectrum_at_planet(stellar_flux, Teff, nw=5000)
        flux_str = stars.photochem_spectrum_string(wv_planet, F_planet, scale_to_planet=False)
        
        # Initialize
        with NamedTemporaryFile('w') as f_species:
            yaml.safe_dump(species_dict, f_species)
            with NamedTemporaryFile('w') as f_settings:
                yaml.safe_dump(settings_dict, f_settings)
                with NamedTemporaryFile('w') as f_flux:
                    f_flux.write(flux_str)
                    f_flux.flush()
                    super().__init__(
                        species_file=f_species.name, 
                        settings_file=f_settings.name, 
                        flux_file=f_flux.name,
                        data_dir=None,
                        double_radiative_grid=True
                    )

        # Ensure bolometric flux is right
        self.rad.set_bolometric_flux(stellar_flux)

def blackbody_spectrum_at_planet(stellar_flux, Teff, nw):

    # Blackbody
    wv_planet = np.logspace(np.log10(0.1), np.log10(100), nw)*1e3 # nm
    F_planet = stars.blackbody(Teff, wv_planet)*np.pi

    # Rescale so that it has the proper stellar flux for the planet
    factor = stellar_flux/stars.energy_in_spectrum(wv_planet, F_planet)
    F_planet *= factor

    return wv_planet, F_planet

# 
class flux():
    def __init__(self, P_bottom=1e6, P_top=1):
        # P_bottom and P_top are both in dynes/cm^2
        self.c = AdiabatClimateSimple()
        self.c.P_top = P_top

        nz = len(c.T)
        P = np.logspace(np.log10(P_bottom), np.log10(P_top),2*nz+1)
        P = np.append(P[0], P[1::2]) # P[0] is surface pressure and P[1:] is the layer-center pressure (dynes/cm^2)

        # 50% of N2 and CO2
        f_i = np.ones((len(P),len(c.species_names)))
        f_i[:] = 0.5

    def objective(self, T):
        # Do RT
        self.c.TOA_fluxes_dry(P, T, f_i)
        # Compute the flux going into each layer
        wrk_sol = self.c.rad.wrk_sol
        wrk_ir = self.c.rad.wrk_ir
        f_total = (wrk_sol.fdn_n - wrk_sol.fup_n) + (wrk_ir.fdn_n - wrk_ir.fup_n)
        F = np.append(f_total[0],(f_total[2::2] - f_total[0:-2:2])) # ergs/cm^2/s going into surface + each layer
        lapse_rate = self.c.lapse_rate # The lapse rate of the given T profile (dlnT/dlnP)
        dry_adiabat = self.c.lapse_rate_intended # dry adiabat lapse rate (dlnT/dlnP)
        return F, lapse_rate, dry_adiabat

    def constraints(self, T):
        pass

    def gradient(self, T, dT=0.001):
        fcn_plus = self.objective(T + dT)[0]
        fcn_minus = self.objective(T - dT)[0]
        return (fcn_plus - fcn_minus) / (2 * dT)
