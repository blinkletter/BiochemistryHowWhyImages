#!/usr/bin/env python
#
# File: OpenMMPerformMinimization.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2026 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using OpenMM, an
# open source molecuar simulation package.
#
# This file is part of MayaChemTools.
#
# MayaChemTools is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# MayaChemTools is distributed in the hope that it will be useful, but without
# any warranty; without even the implied warranty of merchantability of fitness
# for a particular purpose.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MayaChemTools; if not, see <http://www.gnu.org/licenses/> or
# write to the Free Software Foundation Inc., 59 Temple Place, Suite 330,
# Boston, MA, 02111-1307, USA.
#

from __future__ import print_function

import os
import sys
import time
import re

# OpenMM imports...
try:
    import openmm as mm
    import openmm.app
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import OpenMM related module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your OpenMM environment and try again.\n\n")
    sys.exit(1)

# MayaChemTools imports...
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), "..", "lib", "Python"))
try:
    from docopt import docopt
    import MiscUtil
    import OpenMMUtil
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import MayaChemTools module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your MayaChemTools environment and try again.\n\n")
    sys.exit(1)

ScriptName = os.path.basename(sys.argv[0])
Options = {}
OptionsInfo = {}


def main():
    """Start execution of the script."""

    MiscUtil.PrintInfo(
        "\n%s (OpenMM v%s; MayaChemTools v%s; %s): Starting...\n"
        % (ScriptName, mm.Platform.getOpenMMVersion(), MiscUtil.GetMayaChemToolsVersion(), time.asctime())
    )

    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()

    # Retrieve command line arguments and options...
    RetrieveOptions()

    # Process and validate command line arguments and options...
    ProcessOptions()

    # Perform actions required by the script...
    PerformMinimization()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def PerformMinimization():
    """Perform minimization."""

    # Prepare system for simulation...
    System, Topology, Positions = PrepareSystem()

    # Freeze and restraint atoms...
    FreezeRestraintAtoms(System, Topology, Positions)

    # Setup integrator...
    Integrator = SetupIntegrator()

    # Setup simulation...
    Simulation = SetupSimulation(System, Integrator, Topology, Positions)

    # Perform energy minimization...
    PerformEnergyMinimization(Simulation)


def PrepareSystem():
    """Prepare system for simulation."""

    System, Topology, Positions = OpenMMUtil.InitializeSystem(
        OptionsInfo["Infile"],
        OptionsInfo["ForcefieldParams"],
        OptionsInfo["SystemParams"],
        OptionsInfo["WaterBox"],
        OptionsInfo["WaterBoxParams"],
        OptionsInfo["SmallMolFile"],
        OptionsInfo["SmallMolID"],
    )

    # Write out a PDB file for the system...
    MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["InitialPDBOutfile"])
    OpenMMUtil.WritePDBFile(
        OptionsInfo["InitialPDBOutfile"], Topology, Positions, OptionsInfo["OutputParams"]["PDBOutKeepIDs"]
    )

    return (System, Topology, Positions)


def SetupIntegrator():
    """Setup integrator."""

    Integrator = OpenMMUtil.InitializeIntegrator(
        OptionsInfo["IntegratorParams"], OptionsInfo["SystemParams"]["ConstraintErrorTolerance"]
    )

    return Integrator


def SetupSimulation(System, Integrator, Topology, Positions):
    """Setup simulation."""

    Simulation = OpenMMUtil.InitializeSimulation(System, Integrator, Topology, Positions, OptionsInfo["PlatformParams"])

    return Simulation


def PerformEnergyMinimization(Simulation):
    """Perform energy minimization."""

    SimulationParams = OpenMMUtil.SetupSimulationParameters(OptionsInfo["SimulationParams"])

    OutputParams = OptionsInfo["OutputParams"]

    # Setup a local minimization reporter...
    MinimizeReporter = None
    if OutputParams["MinimizationDataStdout"] or OutputParams["MinimizationDataLog"]:
        MinimizeReporter = LocalMinimizationReporter()

    if MinimizeReporter is not None:
        MiscUtil.PrintInfo("\nAdding minimization reporters...")
        if OutputParams["MinimizationDataLog"]:
            MiscUtil.PrintInfo(
                "Adding data log minimization reporter (Steps: %s; File: %s)..."
                % (OutputParams["MinimizationDataSteps"], OutputParams["MinimizationDataLogFile"])
            )
        if OutputParams["MinimizationDataStdout"]:
            MiscUtil.PrintInfo(
                "Adding data stdout minimization reporter (Steps: %s)..." % (OutputParams["MinimizationDataSteps"])
            )
    else:
        MiscUtil.PrintInfo("\nSkipping addition of minimization reporters...")

    MaxSteps = SimulationParams["MinimizationMaxSteps"]

    MaxStepsMsg = "MaxSteps: %s" % ("UntilConverged" if MaxSteps == 0 else MaxSteps)
    ToleranceMsg = "Tolerance: %.2f kcal/mol/A (%.2f kjoules/mol/nm)" % (
        SimulationParams["MinimizationToleranceInKcal"],
        SimulationParams["MinimizationToleranceInJoules"],
    )

    MiscUtil.PrintInfo("\nPerforming energy minimization (%s; %s)..." % (MaxStepsMsg, ToleranceMsg))

    if OutputParams["MinimizationDataStdout"]:
        HeaderLine = SetupMinimizationDataOutHeaderLine()
        print("\n%s" % HeaderLine)

    Simulation.minimizeEnergy(
        tolerance=SimulationParams["MinimizationTolerance"], maxIterations=MaxSteps, reporter=MinimizeReporter
    )

    if OutputParams["MinimizationDataLog"]:
        WriteMinimizationDataLogFile(MinimizeReporter.DataOutValues)

    # Write out minimized structure...
    MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["MinimizedPDBOutfile"])
    OpenMMUtil.WriteSimulationStatePDBFile(
        Simulation, OptionsInfo["MinimizedPDBOutfile"], OptionsInfo["OutputParams"]["PDBOutKeepIDs"]
    )


class LocalMinimizationReporter(mm.MinimizationReporter):
    """Setup a local minimization reporter."""

    (DataSteps, DataOutTypeList, DataOutDelimiter, StdoutStatus) = [None] * 4

    DataOutValues = []
    First = True

    def report(self, Iteration, PositonsList, GradientsList, DataStatisticsMap):
        """Report and track minimization."""

        if self.First:
            # Initialize...
            self.DataSteps = OptionsInfo["OutputParams"]["MinimizationDataSteps"]
            self.DataOutTypeList = OptionsInfo["OutputParams"]["MinimizationDataOutTypeOpenMMNameList"]
            self.DataOutDelimiter = OptionsInfo["OutputParams"]["DataOutDelimiter"]
            self.StdoutStatus = True if OptionsInfo["OutputParams"]["MinimizationDataStdout"] else False

            self.First = False

        if Iteration % self.DataSteps == 0:
            # Setup data values...
            DataValues = []
            DataValues.append("%s" % Iteration)
            for DataType in self.DataOutTypeList:
                DataValue = "%.4f" % DataStatisticsMap[DataType]
                DataValues.append(DataValue)

            # Track data...
            self.DataOutValues.append(DataValues)

            # Print data values...
            if self.StdoutStatus:
                print("%s" % self.DataOutDelimiter.join(DataValues))

        # This method must return a bool. You may return true for early termination.
        return False


def WriteMinimizationDataLogFile(DataOutValues):
    """Write minimization data log file."""

    OutputParams = OptionsInfo["OutputParams"]

    Outfile = OutputParams["MinimizationDataLogFile"]
    OutDelimiter = OutputParams["DataOutDelimiter"]

    MiscUtil.PrintInfo("\nWriting minimization log file %s..." % Outfile)
    OutFH = open(Outfile, "w")

    HeaderLine = SetupMinimizationDataOutHeaderLine()
    OutFH.write("%s\n" % HeaderLine)

    for LineWords in DataOutValues:
        Line = OutDelimiter.join(LineWords)
        OutFH.write("%s\n" % Line)

    OutFH.close()


def SetupMinimizationDataOutHeaderLine():
    """Setup minimization data output header line."""

    LineWords = ["Iteration"]
    for Label in OptionsInfo["OutputParams"]["MinimizationDataOutTypeList"]:
        if re.match("^(SystemEnergy|RestraintEnergy)$", Label, re.I):
            LineWords.append("%s(kjoules/mol)" % Label)
        elif re.match("^RestraintStrength$", Label, re.I):
            LineWords.append("%s(kjoules/mol/nm^2)" % Label)
        else:
            LineWords.append(Label)

    Line = OptionsInfo["OutputParams"]["DataOutDelimiter"].join(LineWords)

    return Line


def FreezeRestraintAtoms(System, Topology, Positions):
    """Handle freezing and restraining of atoms."""

    FreezeAtomList, RestraintAtomList = OpenMMUtil.ValidateAndFreezeRestraintAtoms(
        OptionsInfo["FreezeAtoms"],
        OptionsInfo["FreezeAtomsParams"],
        OptionsInfo["RestraintAtoms"],
        OptionsInfo["RestraintAtomsParams"],
        OptionsInfo["RestraintSpringConstant"],
        OptionsInfo["SystemParams"],
        System,
        Topology,
        Positions,
    )


def ProcessOutfilePrefixParameter():
    """Process outfile prefix paramater."""

    OutfilePrefix = Options["--outfilePrefix"]

    if not re.match("^auto$", OutfilePrefix, re.I):
        OptionsInfo["OutfilePrefix"] = OutfilePrefix
        return

    if OptionsInfo["SmallMolFileMode"]:
        OutfilePrefix = "%s_%s_Complex" % (OptionsInfo["InfileRoot"], OptionsInfo["SmallMolFileRoot"])
    else:
        OutfilePrefix = "%s" % (OptionsInfo["InfileRoot"])

    if re.match("^yes$", Options["--waterBox"], re.I):
        OutfilePrefix = "%s_Solvated" % (OutfilePrefix)

    OptionsInfo["OutfilePrefix"] = OutfilePrefix


def ProcessOutfileNames():
    """Process outfile names."""

    OutputParams = OptionsInfo["OutputParams"]

    OutfileParamName = "MinimizationDataLogFile"
    OutfileParamValue = OutputParams[OutfileParamName]
    if not Options["--overwrite"]:
        if os.path.exists(OutfileParamValue):
            MiscUtil.PrintError(
                'The file specified, %s, for parameter name, %s, using option "--outfileParams" already exist. Use option "--ov" or "--overwrite" and try again. '
                % (OutfileParamValue, OutfileParamName)
            )

    InitialPDBOutfile = "%s_Initial.%s" % (OptionsInfo["OutfilePrefix"], OutputParams["PDBOutfileExt"])
    MinimizedPDBOutfile = "%s_Minimized.%s" % (OptionsInfo["OutfilePrefix"], OutputParams["PDBOutfileExt"])
    for Outfile in [InitialPDBOutfile, MinimizedPDBOutfile]:
        if not Options["--overwrite"]:
            if os.path.exists(Outfile):
                MiscUtil.PrintError(
                    'The file name, %s, generated using option "--outfilePrefix" already exist. Use option "--ov" or "--overwrite" and try again. '
                    % (Outfile)
                )
    OptionsInfo["InitialPDBOutfile"] = InitialPDBOutfile
    OptionsInfo["MinimizedPDBOutfile"] = MinimizedPDBOutfile


def ProcessWaterBoxParameters():
    """Process water box parameters."""

    OptionsInfo["WaterBox"] = True if re.match("^yes$", Options["--waterBox"], re.I) else False
    OptionsInfo["WaterBoxParams"] = OpenMMUtil.ProcessOptionOpenMMWaterBoxParameters(
        "--waterBoxParams", Options["--waterBoxParams"]
    )

    if OptionsInfo["WaterBox"]:
        if OptionsInfo["ForcefieldParams"]["ImplicitWater"]:
            MiscUtil.PrintInfo("")
            MiscUtil.PrintWarning(
                'The value, %s, specified using option "--waterBox" may not be valid for the combination of biopolymer and water forcefields, %s and %s, specified using "--forcefieldParams". You may consider using a valid combination of biopolymer and water forcefields for explicit water during the addition of a water box.'
                % (
                    Options["--waterBox"],
                    OptionsInfo["ForcefieldParams"]["Biopolymer"],
                    OptionsInfo["ForcefieldParams"]["Water"],
                )
            )


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["Infile"])
    OptionsInfo["InfileRoot"] = FileName

    SmallMolFile = Options["--smallMolFile"]
    SmallMolID = Options["--smallMolID"]
    SmallMolFileMode = False
    SmallMolFileRoot = None
    if SmallMolFile is not None:
        FileDir, FileName, FileExt = MiscUtil.ParseFileName(SmallMolFile)
        SmallMolFileRoot = FileName
        SmallMolFileMode = True

    OptionsInfo["SmallMolFile"] = SmallMolFile
    OptionsInfo["SmallMolFileRoot"] = SmallMolFileRoot
    OptionsInfo["SmallMolFileMode"] = SmallMolFileMode
    OptionsInfo["SmallMolID"] = SmallMolID.upper()

    ProcessOutfilePrefixParameter()

    ParamsDefaultInfoOverride = {
        "MinimizationDataSteps": 100,
        "MinimizationDataStdout": False,
        "MinimizationDataLog": True,
    }
    OptionsInfo["OutputParams"] = OpenMMUtil.ProcessOptionOpenMMOutputParameters(
        "--outputParams", Options["--outputParams"], OptionsInfo["OutfilePrefix"], ParamsDefaultInfoOverride
    )
    ProcessOutfileNames()

    OptionsInfo["ForcefieldParams"] = OpenMMUtil.ProcessOptionOpenMMForcefieldParameters(
        "--forcefieldParams", Options["--forcefieldParams"]
    )

    OptionsInfo["FreezeAtoms"] = True if re.match("^yes$", Options["--freezeAtoms"], re.I) else False
    if OptionsInfo["FreezeAtoms"]:
        OptionsInfo["FreezeAtomsParams"] = OpenMMUtil.ProcessOptionOpenMMAtomsSelectionParameters(
            "--freezeAtomsParams", Options["--freezeAtomsParams"]
        )
    else:
        OptionsInfo["FreezeAtomsParams"] = None

    ParamsDefaultInfoOverride = {"Name": Options["--platform"], "Threads": 1}
    OptionsInfo["PlatformParams"] = OpenMMUtil.ProcessOptionOpenMMPlatformParameters(
        "--platformParams", Options["--platformParams"], ParamsDefaultInfoOverride
    )

    OptionsInfo["RestraintAtoms"] = True if re.match("^yes$", Options["--restraintAtoms"], re.I) else False
    if OptionsInfo["RestraintAtoms"]:
        OptionsInfo["RestraintAtomsParams"] = OpenMMUtil.ProcessOptionOpenMMAtomsSelectionParameters(
            "--restraintAtomsParams", Options["--restraintAtomsParams"]
        )
    else:
        OptionsInfo["RestraintAtomsParams"] = None
    OptionsInfo["RestraintSpringConstant"] = float(Options["--restraintSpringConstant"])

    OptionsInfo["SystemParams"] = OpenMMUtil.ProcessOptionOpenMMSystemParameters(
        "--systemParams", Options["--systemParams"]
    )

    OptionsInfo["IntegratorParams"] = OpenMMUtil.ProcessOptionOpenMMIntegratorParameters(
        "--integratorParams",
        Options["--integratorParams"],
        HydrogenMassRepartioningStatus=OptionsInfo["SystemParams"]["HydrogenMassRepartioning"],
    )

    OptionsInfo["SimulationParams"] = OpenMMUtil.ProcessOptionOpenMMSimulationParameters(
        "--simulationParams", Options["--simulationParams"]
    )

    ProcessWaterBoxParameters()

    OptionsInfo["Overwrite"] = Options["--overwrite"]


def RetrieveOptions():
    """Retrieve command line arguments and options."""

    # Get options...
    global Options
    Options = docopt(_docoptUsage_)

    # Set current working directory to the specified directory...
    WorkingDir = Options["--workingdir"]
    if WorkingDir:
        os.chdir(WorkingDir)

    # Handle examples option...
    if "--examples" in Options and Options["--examples"]:
        MiscUtil.PrintInfo(MiscUtil.GetExamplesTextFromDocOptText(_docoptUsage_))
        sys.exit(0)


def ValidateOptions():
    """Validate option values."""

    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "pdb cif")

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--infile"])
    OutfilePrefix = Options["--outfilePrefix"]
    if not re.match("^auto$", OutfilePrefix, re.I):
        if re.match("^(%s)$" % OutfilePrefix, FileName, re.I):
            MiscUtil.PrintError(
                'The value specified, %s, for option "--outfilePrefix" is not valid. You must specify a value different from, %s, the root of infile name.'
                % (OutfilePrefix, FileName)
            )

    if Options["--smallMolFile"] is not None:
        MiscUtil.ValidateOptionFilePath("-l, --smallMolFile", Options["--smallMolFile"])
        MiscUtil.ValidateOptionFileExt("-l, --smallMolFile", Options["--smallMolFile"], "sd sdf")

    SmallMolID = Options["--smallMolID"]
    if len(SmallMolID) != 3:
        MiscUtil.PrintError(
            'The value specified, %s, for option "--smallMolID" is not valid. You must specify a three letter small molecule ID.'
            % (SmallMolID)
        )

    MiscUtil.ValidateOptionTextValue("--freezeAtoms", Options["--freezeAtoms"], "yes no")
    if re.match("^yes$", Options["--freezeAtoms"], re.I):
        if Options["--freezeAtomsParams"] is None:
            MiscUtil.PrintError(
                'No value specified for option "--freezeAtomsParams". You must specify valid values during, yes, value for "--freezeAtoms" option.'
            )

    MiscUtil.ValidateOptionTextValue("-p, --platform", Options["--platform"], "CPU CUDA OpenCL Reference")

    MiscUtil.ValidateOptionTextValue("--restraintAtoms", Options["--restraintAtoms"], "yes no")
    if re.match("^yes$", Options["--restraintAtoms"], re.I):
        if Options["--restraintAtomsParams"] is None:
            MiscUtil.PrintError(
                'No value specified for option "--restraintAtomsParams". You must specify valid values during, yes, value for "--restraintAtoms" option.'
            )

    MiscUtil.ValidateOptionFloatValue("--restraintSpringConstant", Options["--restraintSpringConstant"], {">": 0})

    MiscUtil.ValidateOptionTextValue("--waterBox", Options["--waterBox"], "yes no")


# Setup a usage string for docopt...
_docoptUsage_ = """
OpenMMPerformMinimization.py - Perform an energy minimization

Usage:
    OpenMMPerformMinimization.py [--forcefieldParams <Name,Value,..>] [--freezeAtoms <yes or no>]
                                 [--freezeAtomsParams <Name,Value,..>] [--integratorParams <Name,Value,..>]
                                 [--outputParams <Name,Value,..>] [--outfilePrefix <text>]
                                 [--overwrite] [--platform <text>] [--platformParams <Name,Value,..>]
                                 [--restraintAtoms <yes or no>]
                                 [--restraintAtomsParams <Name,Value,..>] [--restraintSpringConstant <number>]
                                 [--simulationParams <Name,Value,..>] [--smallMolFile <SmallMolFile>] [--smallMolID <text>]
                                 [--systemParams <Name,Value,..>] [--waterBox <yes or no>]
                                 [--waterBoxParams <Name,Value,..>] [-w <dir>] -i <infile>
    OpenMMPerformMinimization.py -h | --help | -e | --examples

Description:
    Perform energy minimization for a macromolecule or a macromolecule in a
    complex with small molecule. You may optionally add a water box and
    freeze/restraint atoms to your system before minimization.

    The input file must contain a macromolecule already prepared for simulation.
    The preparation of the macromolecule for a simulation generally involves the
    following: identification and replacement non-standard residues; addition of
    missing residues; addition of missing heavy atoms; addition of missing
    hydrogens; addition of a water box which is optional.

    In addition, the small molecule input file must contain a molecule already
    prepared for simulation. It must contain  appropriate 3D coordinates relative
    to the macromolecule along with no missing hydrogens.

    The supported macromolecule input file formats are:  PDB (.pdb) and
    CIF (.cif)

    The supported small molecule input file format are : SD (.sdf, .sd)

    Possible outfile prefixes:
         
        <InfileRoot>
        <InfileRoot>_Solvated
        <InfileRoot>_<SmallMolFileRoot>_Complex
        <InfileRoot>_<SmallMolFileRoot>_Complex_Solvated
         
    Possible output files:
         
        <OutfilePrefix>_Initial.<pdb or cif>
        <OutfilePrefix>_Minimized.<pdb or cif>
        <OutfilePrefix>_Minimization.csv
         
Options:
    -e, --examples
        Print examples.
    -f, --forcefieldParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for biopolymer,
        water, and small molecule forcefields.
        
        The supported parameter names along with their default values are
        are shown below:
            
            biopolymer, amber14-all.xml  [ Possible values: Any Valid value ]
            smallMolecule, openff-2.2.1  [ Possible values: Any Valid value ]
            water, auto  [ Possible values: Any Valid value ]
            additional, none [ Possible values: Space delimited list of any
                valid value ]
            
        Possible biopolymer forcefield values:
            
            amber14-all.xml, amber99sb.xml, amber99sbildn.xml, amber03.xml,
            amber10.xml
            charmm36.xml, charmm_polar_2019.xml
            amoeba2018.xml
        
        Possible small molecule forcefield values:
            
            openff-2.2.1, openff-2.0.0, openff-1.3.1, openff-1.2.1,
            openff-1.1.1, openff-1.1.0,...
            smirnoff99Frosst-1.1.0, smirnoff99Frosst-1.0.9,...
            gaff-2.11, gaff-2.1, gaff-1.81, gaff-1.8, gaff-1.4,...
        
        The default water forcefield valus is dependent on the type of the
        biopolymer forcefield as shown below:
            
            Amber: amber14/tip3pfb.xml
            CHARMM: charmm36/water.xml or None for charmm_polar_2019.xml
            Amoeba: None (Explicit)
            
        Possible water forcefield values:
            
            amber14/tip3p.xml, amber14/tip3pfb.xml, amber14/spce.xml,
            amber14/tip4pew.xml, amber14/tip4pfb.xml,
            charmm36/water.xml, charmm36/tip3p-pme-b.xml,
            charmm36/tip3p-pme-f.xml, charmm36/spce.xml,
            charmm36/tip4pew.xml, charmm36/tip4p2005.xml,
            charmm36/tip5p.xml, charmm36/tip5pew.xml,
            implicit/obc2.xml, implicit/GBn.xml, implicit/GBn2.xml,
            amoeba2018_gk.xml (Implict water)
            None (Explicit water for amoeba)
        
        The additional forcefield value is a space delimited list of any valid
        forcefield values and is passed on to the OpenMMForcefields
        SystemGenerator along with the specified forcefield  values for
        biopolymer, water, and mall molecule. Possible additional forcefield
        values are:
            
            amber14/DNA.OL15.xml amber14/RNA.OL3.xml
            amber14/lipid17.xml amber14/GLYCAM_06j-1.xml
            ... ... ...
            
        You may specify any valid forcefield names supported by OpenMM. No
        explicit validation is performed.
    --freezeAtoms <yes or no>  [default: no]
        Freeze atoms during energy minimization. The specified atoms are kept
        completely fixed by setting their masses to zero. Their positions do not
        change during  energy minimization.
    --freezeAtomsParams <Name,Value,..>
        A comma delimited list of parameter name and value pairs for freezing
        atoms during energy minimization. You must specify these parameters for 'yes'
        value of '--freezeAtoms' option.
        
        The supported parameter names along with their default values are
        are shown below:
            
            selection, none [ Possible values: CAlphaProtein, Ions, Ligand,
                Protein, Residues, or Water ]
            selectionSpec, auto [ Possible values: A space delimited list of
                residue names ]
            negate, no [ Possible values: yes or no ]
            
        A brief description of parameters is provided below:
            
            selection: Atom selection to freeze.
            selectionSpec: A space delimited list of residue names for
                selecting atoms to freeze. You must specify its value during
                'Ligand' and 'Protein' value for 'selection'. The default values
                are automatically set for 'CAlphaProtein', 'Ions', 'Protein',
                and 'Water' values of 'selection' as shown below:
                
                CAlphaProtein: List of stadard protein residues from pdbfixer
                    for selecting CAlpha atoms.
                Ions: Li Na K Rb Cs Cl Br F I
                Water: HOH
                Protein: List of standard protein residues from pdbfixer.
                
            negate: Negate atom selection match to select atoms for freezing.
            
        In addition, you may specify an explicit space delimited list of residue
        names using 'selectionSpec' for any 'selection". The specified residue
        names are appended to the appropriate default values during the
        selection of atoms for freezing.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name containing a macromolecule.
    --integratorParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for integrator
        to setup the system for local energy minimization. No MD simulation is
        performed.
        
        The supported parameter names along with their default values are
        are shown below:
            
            temperature, 300.0 [ Units: kelvin ]
            
    --outfilePrefix <text>  [default: auto]
        File prefix for generating the names of output files. The default value
        depends on the names of input files for macromolecule and small molecule
        along with the type of statistical ensemble and the nature of the solvation.
        
        The possible values for outfile prefix are shown below:
            
            <InfileRoot>
            <InfileRoot>_Solvated
            <InfileRoot>_<SmallMolFileRoot>_Complex
            <InfileRoot>_<SmallMolFileRoot>_Complex_Solvated
            
    --outputParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
       output during energy minimization..
        
        The supported parameter names along with their default values are
        are shown below:
            
            minimizationDataSteps, 100
            minimizationDataStdout, no  [ Possible values: yes or no ]
            minimizationDataLog, yes  [ Possible values: yes or no ]
            minimizationDataLogFile, auto  [ Default:
                <OutfilePrefix>_MinimizationOut.csv ]
            minimizationDataOutType, auto [ Possible values: A space delimited
                list of valid parameter names.  Default: SystemEnergy
                RestraintEnergy MaxConstraintError.
                Other valid names: RestraintStrength ]
            
            pdbOutFormat, PDB  [ Possible values: PDB or CIF ]
            pdbOutKeepIDs, yes  [ Possible values: yes or no ]
            
        A brief description of parameters is provided below:
            
            minimizationDataSteps: Frequency of writing data to stdout
                and log file.
            minimizationDataStdout: Write data to stdout.
            minimizationDataLog: Write data to log file.
            minimizationDataLogFile: Data log fie name.
            minimizationDataOutType: Type of data to write to stdout
                and log file.
            
            pdbOutFormat: Format of output PDB files.
            pdbOutKeepIDs: Keep existing chain and residue IDs.
            
    --overwrite
        Overwrite existing files.
    -p, --platform <text>  [default: CPU]
        Platform to use for running MD simulation. Possible values: CPU, CUDA,
       OpenCL, or Reference.
    --platformParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs to configure
        platform for running energy minimization calculations..
        
        The supported parameter names along with their default values for
        different platforms are shown below:
            
            CPU:
            
            threads, 1  [ Possible value: >= 0 or auto.  The value of 'auto'
                or zero implies the use of all available CPUs for threading. ]
            
            CUDA:
            
            deviceIndex, auto  [ Possible values: 0, '0 1' etc. ]
            deterministicForces, auto [ Possible values: yes or no ]
            precision, single  [ Possible values: single, double, or mix ]
            tempDirectory, auto [ Possible value: DirName ]
            useBlockingSync, auto [ Possible values: yes or no ]
            useCpuPme, auto [ Possible values: yes or no ]
            
            OpenCL:
            
            deviceIndex, auto  [ Possible values: 0, '0 1' etc. ]
            openCLPlatformIndex, auto  [ Possible value: Number]
            precision, single  [ Possible values: single, double, or mix ]
            useCpuPme, auto [ Possible values: yes or no ]
            
        A brief description of parameters is provided below:
            
            CPU:
            
            threads: Number of threads to use for simulation.
            
            CUDA:
            
            deviceIndex: Space delimited list of device indices to use for
                calculations.
            deterministicForces: Generate reproducible results at the cost of a
                small decrease in performance.
            precision: Number precision to use for calculations.
            tempDirectory: Directory name for storing temporary files.
            useBlockingSync: Control run-time synchronization between CPU and
                GPU.
            useCpuPme: Use CPU-based PME implementation.
            
            OpenCL:
            
            deviceIndex: Space delimited list of device indices to use for
                simulation.
            openCLPlatformIndex: Platform index to use for calculations.
            precision: Number precision to use for calculations.
            useCpuPme: Use CPU-based PME implementation.
            
    --restraintAtoms <yes or no>  [default: no]
        Restraint atoms during energy minimization. The motion of specified atoms is
        restricted by adding a harmonic force that binds them to their starting
        positions. The atoms are not completely fixed unlike freezing of atoms.
        Their motion, however, is restricted and they are not able to move far away
        from their starting positions during energy minimization.
    --restraintAtomsParams <Name,Value,..>
        A comma delimited list of parameter name and value pairs for restraining
        atoms during energy minimization. You must specify these parameters for 'yes'
        value of '--restraintAtoms' option.
        
        The supported parameter names along with their default values are
        are shown below:
            
            selection, none [ Possible values: CAlphaProtein, Ions, Ligand,
                Protein, Residues, or Water ]
            selectionSpec, auto [ Possible values: A space delimited list of
                residue names ]
            negate, no [ Possible values: yes or no ]
            
        A brief description of parameters is provided below:
            
            selection: Atom selection to restraint.
            selectionSpec: A space delimited list of residue names for
                selecting atoms to restraint. You must specify its value during
                'Ligand' and 'Protein' value for 'selection'. The default values
                are automatically set for 'CAlphaProtein', 'Ions', 'Protein',
                and 'Water' values of 'selection' as shown below:
                
                CAlphaProtein: List of stadard protein residues from pdbfixer
                    for selecting CAlpha atoms.
                Ions: Li Na K Rb Cs Cl Br F I
                Water: HOH
                Protein: List of standard protein residues from pdbfixer.
                
            negate: Negate atom selection match to select atoms for freezing.
            
        In addition, you may specify an explicit space delimited list of residue
        names using 'selectionSpec' for any 'selection". The specified residue
        names are appended to the appropriate default values during the
        selection of atoms for restraining.
    --restraintSpringConstant <number>  [default: 2.5]
        Restraint spring constant for applying external restraint force to restraint
        atoms relative to their initial positions during 'yes' value of '--restraintAtoms'
        option. Default units: kcal/mol/A**2. The default value, 2.5, corresponds to
        1046.0 kjoules/mol/nm**2. The default value is automatically converted into
        units of kjoules/mol/nm**2 before its usage.
    --simulationParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for simulation.
        
        The supported parameter names along with their default values are
        are shown below:
            
            minimizationMaxSteps, auto  [ Possible values: >= 0. The value of
                zero implies until the minimization is converged. ]
            minimizationTolerance, 0.24  [ Units: kcal/mol/A. The default value
                0.24, corresponds to OpenMM default of value of 10.04
                kjoules/mol/nm. It is automatically converted into OpenMM
                default units before its usage. ]
            
        A brief description of parameters is provided below:
            
            minimizationMaxSteps: Maximum number of minimization steps. The
                value of zero implies until the minimization is converged.
            minimizationTolerance: Energy convergence tolerance during
                minimization.
            
    -s, --smallMolFile <SmallMolFile>
        Small molecule input file name. The macromolecue and small molecule are
        merged for simulation and the complex is written out to a PDB file.
    --smallMolID <text>  [default: LIG]
        Three letter small molecule residue ID. The small molecule ID corresponds
        to the residue name of the small molecule and is written out to a PDB file
        containing the complex.
    --systemParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs to configure
        a system for energy minimization. No MD simulation is performed.
        
        The supported parameter names along with their default values are
        are shown below:
            
            constraints, BondsInvolvingHydrogens [ Possible values: None,
                WaterOnly, BondsInvolvingHydrogens, AllBonds, or
                AnglesInvolvingHydrogens ]
            constraintErrorTolerance, 0.000001
            ewaldErrorTolerance, 0.0005
            
            nonbondedMethodPeriodic, PME [ Possible values: NoCutoff,
                CutoffNonPeriodic, or PME ]
            nonbondedMethodNonPeriodic, NoCutoff [ Possible values:
                NoCutoff or CutoffNonPeriodic]
            nonbondedCutoff, 1.0 [ Units: nm ]
            
            hydrogenMassRepartioning, yes [ Possible values: yes or no ]
            hydrogenMass, 1.5 [ Units: amu]
            
            removeCMMotion, yes [ Possible values: yes or no ]
            rigidWater, auto [ Possible values: yes or no. Default: 'No' for
                'None' value of constraints; Otherwise, yes ]
            
        A brief description of parameters is provided below:
            
            constraints: Type of system constraints to use for simulation. These
                constraints are different from freezing and restraining of any
                atoms in the system.
            constraintErrorTolerance: Distance tolerance for constraints as a
                fraction of the constrained distance.
            ewaldErrorTolerance: Ewald error tolerance for a periodic system.
            
            nonbondedMethodPeriodic: Nonbonded method to use during the
                calculation of long range interactions for a periodic system.
            nonbondedMethodNonPeriodic: Nonbonded method to use during the
                calculation of long range interactions for a non-periodic system.
            nonbondedCutoff: Cutoff distance to use for long range interactions
                in both perioidic non-periodic systems.
            
            hydrogenMassRepartioning: Use hydrogen mass repartioning. It
                increases the mass of the hydrogen atoms attached to the heavy
                atoms and decreasing the mass of the bonded heavy atom to
                maintain constant system mass. This allows the use of larger
                integration step size (4 fs) during a simulation.
            hydrogenMass: Hydrogen mass to use during repartioning.
            
            removeCMMotion: Remove all center of mass motion at every time step.
            rigidWater: Keep water rigid during a simulation. This is determined
                automatically based on the value of 'constraints' parameter.
            
    --waterBox <yes or no>  [default: no]
        Add water box.
    --waterBoxParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for adding
        a water box.
        
        The supported parameter names along with their default values are
        are shown below:
            
            model, tip3p [ Possible values: tip3p, spce, tip4pew, tip5p or
                swm4ndp ]
            mode, Padding  [ Possible values: Size or Padding ]
            padding, 1.0
            size, None  [ Possible value: xsize ysize zsize ]
            shape, cube  [ Possible values: cube, dodecahedron, or octahedron ]
            ionPositive, Na+ [ Possible values: Li+, Na+, K+, Rb+, or Cs+ ]
            ionNegative, Cl- [ Possible values: Cl-, Br-, F-, or I- ]
            ionicStrength, 0.0
            
        A brief description of parameters is provided below:
            
            model: Water model to use for adding water box. The van der
                Waals radii and atomic charges are determined using the
                specified water forcefield. You must specify an appropriate
                water forcefield. No validation is performed.
            mode: Specify the size of the waterbox explicitly or calculate it
                automatically for a macromolecule along with adding padding
                around ther macromolecule.
            padding: Padding around a macromolecule in nanometers for filling
                box with water. It must be specified during 'Padding' value of
                'mode' parameter.
            size: A space delimited triplet of values corresponding to water
                size in nanometers. It must be specified during 'Size' value of
                'mode' parameter.
            ionPositive: Type of positive ion to add during the addition of a
                water box.
            ionNegative: Type of negative ion to add during the addition of a
                water box.
            ionicStrength: Total concentration of both positive and negative
                ions to add excluding the ions added to neutralize the system
                during the addition of a water box.
            
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To perform energy minimization for a macromolecule in a PDB file until the
    energy is converged, writing information to log file every 100 steps and 
    generate a PDB file for the minimized system, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb

    To run the first example for writing information to both stdout and log file
     every 100 steps and generate various output files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb --outputParams
          "minimizationDataStdout, yes"

    To run the first example for performing OpenMM simulation using multi-
    threading employing all available CPUs on your machine and generate various
    output files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb
          --platformParams "threads,0"

    To run the first example for performing OpenMM simulation using CUDA platform
    on your machine and generate various output files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb -p CUDA

    To run the second example for a marcomolecule in a complex with a small
    molecule and generate various output files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb -s Sample13Ligand.sdf
          --platformParams "threads,0"

    To run the second example by adding a water box to the system and generate
    various output files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb --waterBox yes
          --platformParams "threads,0"

    To run the second example for a marcomolecule in a complex with a small
    molecule by adding a water box to the system and generate various output
    files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb -s Sample13Ligand.sdf
          --waterBox yes --platformParams "threads,0"

    To run the second example by freezing CAlpha atoms in a macromolecule without
    using any system constraints to avoid any issues with the freezing of the same atoms
    and generate various output files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb
          --freezeAtoms yes --freezeAtomsParams "selection,CAlphaProtein"
          --systemParams "constraints, None"
          --platformParams "threads,0"

        % OpenMMPerformMinimization.py -i Sample13.pdb
          --freezeAtoms yes --freezeAtomsParams "selection,CAlphaProtein"
          --systemParams "constraints, None"
          --platformParams "threads,0" --waterBox yes

    To run the second example by restrainting CAlpha atoms in a macromolecule and
    and generate various output files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb
          --restraintAtomsParams "selection,CAlphaProtein"
          --platformParams "threads,0"

        % OpenMMPerformMinimization.py -i Sample13.pdb
          --restraintAtomsParams "selection,CAlphaProtein"
          --platformParams "threads,0"
          --waterBox yes

    To run the second example by specifying explict values for various parametres
    and generate various output files, type:

        % OpenMMPerformMinimization.py -i Sample13.pdb
          -f " biopolymer,amber14-all.xml,smallMolecule, openff-2.2.1,
          water,amber14/tip3pfb.xml" --waterBox yes
          --outputParams "minimizationDataSteps, 100, minimizationDataStdout,
          yes,minimizationDataLog,yes"
          -p CPU --platformParams "threads,0"
          --simulationParams "minimizationMaxSteps,500,
          minimizationTolerance, 0.24"
          --systemParams "constraints,BondsInvolvingHydrogens,
          nonbondedMethodPeriodic,PME,nonbondedMethodNonPeriodic,NoCutoff,
          hydrogenMassRepartioning, yes"

Author:
    Manish Sud(msud@san.rr.com)

See also:
    OpenMMPrepareMacromolecule.py, OpenMMPerformMDSimulation.py,
    OpenMMPerformSimulatedAnnealing.py

Copyright:
    Copyright (C) 2026 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using OpenMM, an
    open source molecuar simulation package.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
