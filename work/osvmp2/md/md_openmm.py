import os
import sys
from simtk import openmm
from osvmp2.mm.xyz2pdb import xyz_to_pdb


def get_unit_openmm(unit_type, unit_default):
    unit = os.environ.get(unit_type, unit_default)
    return getattr(openmm.unit, unit)
    

class md_parameters():
    def __init__(self):
        self.xyz_name = sys.argv[1]
        self.len_side = get_SideLength(self.xyz_name)*10
        self.cen_atom = get_cen_atom(self.xyz_name)
        self.chkfile = os.environ.get("chkfile", None)
        self.output_opt = os.environ.get("output_opt", 'prop')
        #Set up MD units
        self.time_unit =get_unit_openmm("time_unit", 'femtosecond')
        self.temp_unit = get_unit_openmm("temp_unit", 'kelvin')
        self.potential_unit = get_unit_openmm("potential_unit", 'electronvolt')
        self.press_unit = get_unit_openmm("press_unit", 'bar')
        self.cell_units = get_unit_openmm("cell_units", 'angstrom')
        self.force_unit = get_unit_openmm("force_unit", 'piconewton')

        #Set up MD parameters
        self.verbose = int(os.environ.get("verbose", 5))
        self.stride = os.environ.get("stride", '10')
        self.total_steps = os.environ.get("total_steps", 10000)
        self.temperature = os.environ.get("temperature", 300)
        self.pressure = os.environ.get("pressure", 10)
        self.if_fixatoms = bool(int(os.environ.get("if_fixatoms", 0)))
        self.fixatoms = os.environ.get("fixatoms", None)
        if self.if_fixatoms and self.fixatoms is None:
            self.fixatoms = [self.cen_atom]

        self.fixcom = bool(int(os.environ.get("fixcom", 1)))
        self.dyn_mode = os.environ.get("dyn_mode", 'nvt')
        self.temp_ensem = bool(int(os.environ.get("temp_ensem", 0)))
        if self.dyn_mode.lower() in ['nvt', 'npt']:
            self.temp_ensem = True
        self.therm_mode = os.environ.get("therm_mode", 'pile_g')
        self.baro_mode = os.environ.get("baro_mode", 'isotropic')
        self.tau = os.environ.get("tau", 100)
        self.time_step = os.environ.get("time_step", 1)
        #For now 
        if self.dyn_mode.lower() == "nvt":
            #TODO friction
            self.integrator = openmm.LangevinMiddleIntegrator(self.temperature*self.temp_unit, 
                                                              1/openmm.unit.picosecond, 
                                                              self.time_step*self.time_unit)
        elif self.dyn_mode.lower() == "nve":
            self.integrator = openmm.VerletIntegrator(self.time_step*self.time_unit)
        else:
            raise NotImplementedError("Ensemble %s is not implemented"%self.dyn_mode)
        self.system = self.forcefield.createSystem(topology,
                                            nonbondedMethod=openmm.app.NoCutoff,
                                            constraints=None, rigidWater=False)