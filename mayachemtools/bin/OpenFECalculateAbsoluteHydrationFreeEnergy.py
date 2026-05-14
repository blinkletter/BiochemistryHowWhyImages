#!/usr/bin/env python
#
# File: OpenFECalculateAbsoluteHydrationFreeEnergy.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2026 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using OpenFE, an
# open source package for alchemical free energy calculations.
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
import logging
import pathlib
import pandas as pd

# OpenFE imports...
try:
    import openfe
    from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import OpenFE related module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your OpenFE environment and try again.\n\n")
    sys.exit(1)

# RDKit imports...
try:
    from rdkit import rdBase
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import RDKit module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your RDKit environment and try again.\n\n")
    sys.exit(1)

# MayaChemTools imports...
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), "..", "lib", "Python"))
try:
    from docopt import docopt
    import MiscUtil
    import OpenFEUtil
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
        "\n%s (OpenFE v%s; OpenMM v%s; RDKit v%s; MayaChemTools v%s; %s): Starting...\n"
        % (
            ScriptName,
            openfe.version("openfe"),
            openfe.version("openmm"),
            rdBase.rdkitVersion,
            MiscUtil.GetMayaChemToolsVersion(),
            time.asctime(),
        )
    )

    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()

    # Retrieve command line arguments and options...
    RetrieveOptions()

    if Options["--list"]:
        ProcessListOption()
    else:
        # Process and validate command line arguments and options...
        ProcessOptions()

        # Perform actions required by the script...
        CalculateAbsoluteHydrationFreeEnergy()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def CalculateAbsoluteHydrationFreeEnergy():
    """Calculate absolute hydration free energy."""

    # Process input file...
    InMols = ProcessInputFile()

    # Process molecule names...
    Mols = ProcessMoleculeNames(InMols)

    # Check for miising partial charges...
    CheckMissingPartialCharges(Mols)

    # Initialize AHFE protocols...
    AHFEProtocol = InitializeAbsoluteSolvationProtocol()

    # Initialize solvent...
    Solvent = InitializeSolventComponent()

    # Setup transformations...
    MolTransformations = SetupTransformations(Mols, Solvent, AHFEProtocol)

    # Setup protocol DAGs...
    MolProtocolDAGs = SetupProtocolDAGs(MolTransformations)

    # Execute protocol DAGs and gather results...
    MolProtocolResults = ExecuteProtocolDAGsAndGatherResults(MolTransformations, MolProtocolDAGs)

    # Process protocol results...
    ProcessProtocolResults(MolTransformations, MolProtocolResults)


def InitializeAbsoluteSolvationProtocol():
    """Initialize absolute solvation protocol."""

    MiscUtil.PrintInfo("\nInitializing absolute solvation protocol...")

    AHFESettings = OpenFEUtil.SetupAbsoluteHydrationFreeEnergySettings("-a, --ahfeParams", OptionsInfo["AHFEParams"])
    AHFEProtocol = OpenFEUtil.InitializeAbsoluteSolvationFreeEngeryProtocol(AHFESettings)

    return AHFEProtocol


def InitializeSolventComponent():
    """Initialize solvent component."""

    SolventParams = OptionsInfo["SolventParams"]
    MiscUtil.PrintInfo(
        "\nInitializing solvent component (PositiveIon: %s; NegativeIon: %s; Neutralize: %s; IonConcentration: %s)..."
        % (
            SolventParams["PositiveIon"],
            SolventParams["NegativeIon"],
            SolventParams["Neutralize"],
            SolventParams["IonConcentration"],
        )
    )

    Solvent = OpenFEUtil.InitializeSolventComponent(SolventParams)

    return Solvent


def SetupTransformations(Mols, Solvent, AHFEProtocol):
    """Set up transformations for molecules."""

    MiscUtil.PrintInfo("\nSetting up transformations (Count: %s)..." % (len(Mols)))

    MolTransformations = []

    for Mol in Mols:
        # Setup a chemical system for a molecule fully interacting in the solvent...
        MolSolventSystem = OpenFEUtil.InitializeChemicalSystem(
            SmallMol=Mol, MacroMol=None, Solvent=Solvent, Name="%s_Solvent" % Mol.name
        )

        # Setup a system for a molecule fully decoupled in the solvent: Only need to use the solvent....
        SolventOnlySystem = OpenFEUtil.InitializeChemicalSystem(
            SmallMol=None, MacroMol=None, Solvent=Solvent, Name="Solvent_Only"
        )

        # Setup a transformation for absolute solvation protocol from MolSolvent to SolventOnly
        # as the vacuum modelling is automatically handled by AbsoluteSolvationProtocol...
        TransformationName = "%s_AbsoluteSolvation" % (Mol.name)
        MolTransformation = OpenFEUtil.InitializeTransformation(
            StateA=MolSolventSystem,
            StateB=SolventOnlySystem,
            Mapping=None,
            Protocol=AHFEProtocol,
            Name=TransformationName,
            Validate=False,
        )

        MolTransformations.append(MolTransformation)

    # Write out transformatios...
    WriteTransformations(MolTransformations)

    return MolTransformations


def WriteTransformations(MolTransformations):
    """Write out transformations."""

    TransformationsOutDirPath = pathlib.Path(OptionsInfo["TransformationsOutDirPath"])

    MiscUtil.PrintInfo(
        "Writing transformations files (Files: *.json; Count: %s; Subdirectory: %s)..."
        % (len(MolTransformations), OptionsInfo["TransformationsOutDir"])
    )

    for Transformation in MolTransformations:
        TransformationFilePath = TransformationsOutDirPath.joinpath("%s.json" % Transformation.name)
        Transformation.dump(TransformationFilePath)


def SetupProtocolDAGs(MolTransformations):
    """Setup protocol Directed Acyclic Graphs (DAGs) for each transformation to
    to perform calculations.
    """

    MiscUtil.PrintInfo("\nSetting up protocol DAGs (Count: %s)..." % len(MolTransformations))

    MolProtocolDAGs = []
    for Transformation in MolTransformations:
        ProtocolDAG = OpenFEUtil.InitializeProtocolDAG(Transformation, Name=Transformation.name)
        MolProtocolDAGs.append(ProtocolDAG)

    return MolProtocolDAGs


def ExecuteProtocolDAGsAndGatherResults(MolTransformations, MolProtocolDAGs):
    """Execute protocol DAGs and gather results."""

    ResultsSharedOutDirPath = OptionsInfo["ResultsOutDirPath"]
    ResultsScratchOutDirPath = OptionsInfo["ResultsScratchOutDirPath"]
    ExecuteDAGParams = OptionsInfo["ExecuteDAGParams"]

    MolProtocolResults = OpenFEUtil.ExecuteProtocolDAGsAndGatherResults(
        MolTransformations,
        MolProtocolDAGs,
        ResultsSharedOutDirPath,
        ResultsScratchOutDirPath,
        KeepShared=ExecuteDAGParams["KeepShared"],
        KeepScratch=ExecuteDAGParams["KeepScratch"],
        NRetries=ExecuteDAGParams["NRetries"],
        WriteResults=True,
    )

    return MolProtocolResults


def ProcessProtocolResults(MolTransformations, MolProtocolResults):
    """Process protocol results."""

    ResultFileParams = OptionsInfo["ResultFileParams"]

    ResultFile = "%s_AHFE_Results.%s" % (OptionsInfo["OutfilePrefix"], ResultFileParams["Ext"])
    ResultFilePath = os.path.join(OptionsInfo["OutfileDirPath"], ResultFile)
    MiscUtil.PrintInfo("\nWriting %s..." % ResultFile)

    Precision = ResultFileParams["Precision"]

    ResultData = []
    for Index in range(0, len(MolProtocolResults), 1):
        MolProtocolResult = MolProtocolResults[Index]

        # Setup mol name using transformation...
        MolTransformation = MolTransformations[Index]
        Mol = MolTransformation.stateA.components["ligand"]
        MolName = Mol.name

        if MolProtocolResult is None:
            DeltaGHydration = "NA"
            DeltaGHydrationUncertainty = "NA"
        else:
            # Setup hydration value without the units...
            DeltaGHydration = MolProtocolResult.get_estimate()
            DeltaGHydration = "%.*f" % (Precision, DeltaGHydration.m)

            # Setup uncertainty value without the units...
            DeltaGHydrationUncertainty = MolProtocolResult.get_uncertainty()
            DeltaGHydrationUncertainty = "%.*f" % (Precision, DeltaGHydrationUncertainty.m)

        ResultData.append([MolName, DeltaGHydration, DeltaGHydrationUncertainty])

    ResultDF = pd.DataFrame(ResultData, columns=["MolName", "AHFE DeltaG (kcal/mol)", "Uncertainty (kcal/mol)"])
    ResultDF.to_csv(ResultFilePath, sep=ResultFileParams["Delim"], lineterminator="\n", index=False)


def ProcessInputFile():
    """Process input file."""

    # Read small molecule input file...
    MiscUtil.PrintInfo("\nReading small molecule file %s..." % OptionsInfo["Infile"])
    Mols, MolCount, ValidMolCount = OpenFEUtil.ReadAndValidateMolecules(
        OptionsInfo["InfilePath"], **OptionsInfo["InfileParams"]
    )

    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount))

    if ValidMolCount == 0:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("No valid molecules found in small molecule input file.\n")

    return Mols


def ProcessMoleculeNames(Mols):
    """Process molecule names."""

    SpecifiedMols = []
    if OptionsInfo["FirstMoleculeMode"]:
        SpecifiedMols.append(Mols[0])
    elif OptionsInfo["AllMoleculesMode"]:
        SpecifiedMols = Mols
    elif OptionsInfo["MoleculesNamesMode"]:
        SpecifiedMols = OpenFEUtil.ProcessMoleculeNames(Mols, OptionsInfo["MoleculeNamesList"])

    return SpecifiedMols


def CheckMissingPartialCharges(Mols):
    """Check missing partial charges for small molecules."""

    MiscUtil.PrintInfo("\nChecking missing partial charges for small molecules...")

    MissingChargesMolCount = OpenFEUtil.GetMissingPartialChargesMolCount(Mols)
    MiscUtil.PrintInfo("Number of molecules with missing partial charges: %s" % MissingChargesMolCount)

    if MissingChargesMolCount == 0:
        return

    if re.match("^Stop$", OptionsInfo["MissingChargeMode"], re.I):
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError(
            'The small molecule input file contains molecules with missing partial charges. The execution of the script has been terminated for "Stop" value of "--missingChargedMode" option. You may continue the execution of the script by specifying "Calculate" value for "--missingChargedMode" option.\n\nThe missing charges will be automatically calculated by OpenFE AbsoluteSolvationProtocol module during the calculation of AHFE. You may control the calculation of partial charges by specifying values for partialCharge* parameters using "--ahfeParams" option.  Alternatively, you may employ the OpenFECalculatePartialCharges.py script to calculate partial charges and use the small molecule input file containing charges to calculate AHFE.\n'
        )
    else:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintWarning(
            'The small molecule input file contains molecules with missing partial charges. The missing charges will be automatically calculated by OpenFE AbsoluteSolvationProtocol module during the calculation of AHFE. You may control the calculation of partial charges by specifying values for partialCharge* parameters using "--ahfeParams" option. Alternatively, you may employ the OpenFECalculatePartialCharges.py script to calculate partial charges and use the small molecule input file containing charges to calculate AHFE.\n'
        )


def ProcessMoleculeNamesOption():
    """Process molecule names Option."""

    OptionsInfo["MoleculeNames"] = Options["--moleculeNames"]
    OptionsInfo["MoleculeNamesList"] = None

    if OptionsInfo["MoleculeNames"] is None:
        return

    MoleculeNamesList = []
    for MoleculeName in OptionsInfo["MoleculeNames"].split(","):
        MoleculeNamesList.append(MoleculeName.strip())

    OptionsInfo["MoleculeNamesList"] = MoleculeNamesList


def ProcessOutfilePrefixOption():
    """Process outfile prefix option."""

    OutfilePrefix = Options["--outfilePrefix"]

    if re.match("^auto$", OutfilePrefix, re.I):
        OutfilePrefix = OptionsInfo["InfileRoot"]

    OptionsInfo["OutfilePrefix"] = OutfilePrefix


def ProcessOutfileDirOption():
    """Process outfile directory Option."""

    # Setup output directory...
    OutfileDir = Options["--outfileDir"]
    OutfileDirPath = os.path.abspath(OutfileDir)
    if not os.path.exists(OutfileDir):
        MiscUtil.PrintInfo("\nCreating output directory %s..." % (OutfileDir))
        os.mkdir(OutfileDirPath)
    OptionsInfo["OutfileDir"] = OutfileDir
    OptionsInfo["OutfileDirPath"] = OutfileDirPath

    # Setup a transformations subdirectory...
    TransformationsOutDir = "Transformations"
    TransformationsOutDirPath = os.path.join(OptionsInfo["OutfileDirPath"], TransformationsOutDir)
    if not os.path.exists(TransformationsOutDirPath):
        os.mkdir(TransformationsOutDirPath)
    OptionsInfo["TransformationsOutDir"] = TransformationsOutDir
    OptionsInfo["TransformationsOutDirPath"] = TransformationsOutDirPath

    # Setup a results subdirectory...
    ResultsOutDir = "Results"
    ResultsOutDirPath = os.path.join(OptionsInfo["OutfileDirPath"], ResultsOutDir)
    if not os.path.exists(ResultsOutDirPath):
        os.mkdir(ResultsOutDirPath)
    OptionsInfo["ResultsOutDir"] = ResultsOutDir
    OptionsInfo["ResultsOutDirPath"] = ResultsOutDirPath

    # Use results subdirectory for scratch results...
    OptionsInfo["ResultsScratchOutDir"] = ResultsOutDir
    OptionsInfo["ResultsScratchOutDirPath"] = ResultsOutDirPath


def ProcessListOption():
    """Process list protocol settings option."""

    AHFESettings = AbsoluteSolvationProtocol.default_settings()

    MiscUtil.PrintInfo("\nListing AHFE settings...")
    OpenFEUtil.ListOpenFESettings(AHFESettings)


def ConfigureLogging():
    """Configure logging."""

    OptionsInfo["LoggingLevel"] = Options["--loggingLevel"]

    if re.match("^Error$", OptionsInfo["LoggingLevel"], re.I):
        LoggingLevel = logging.ERROR
    elif re.match("^Warning$", OptionsInfo["LoggingLevel"], re.I):
        LoggingLevel = logging.WARNING
    else:
        LoggingLevel = logging.INFO

    logging.basicConfig(format="%(levelname)s: %(message)s", level=LoggingLevel)

    # Turn warnings issued by warnings.warn() into log message to avoid display
    # of a stack trace...
    logging.captureWarnings(True)


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    # Configure logging...
    ConfigureLogging()

    OptionsInfo["Infile"] = Options["--infile"]
    OptionsInfo["InfilePath"] = os.path.abspath(OptionsInfo["Infile"])
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["Infile"])
    OptionsInfo["InfileRoot"] = FileName

    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters(
        "--infileParams",
        Options["--infileParams"],
        InfileName=Options["--infile"],
        ParamsDefaultInfo=ParamsDefaultInfoOverride,
    )

    ParamsDefaultInfoOverride = {"SolventEngineComputePlatform": "CPU", "VacuumEngineComputePlatform": "CPU"}
    ParamsDefaultInfoOverride = None
    OptionsInfo["AHFEParams"] = OpenFEUtil.ProcessOptionOpenFEAbsoluteHydrationFreeEnergyParameters(
        "--ahfeParams", Options["--ahfeParams"], ParamsDefaultInfo=ParamsDefaultInfoOverride
    )

    OptionsInfo["ExecuteDAGParams"] = OpenFEUtil.ProcessOptionOpenFEExecuteDAGParameters(
        "--executeDAGParams", Options["--executeDAGParams"]
    )
    OptionsInfo["LoggingLevel"] = Options["--loggingLevel"]

    OptionsInfo["Mode"] = OpenFEUtil.ProcessOptionOpenFEAbsoluteFreeEnergyMode("-m, --mode", Options["--mode"])
    OptionsInfo["FirstMoleculeMode"] = True if re.match("^FirstMolecule$", OptionsInfo["Mode"], re.I) else False
    OptionsInfo["AllMoleculesMode"] = True if re.match("^AllMolecules$", OptionsInfo["Mode"], re.I) else False
    OptionsInfo["MoleculesNamesMode"] = True if re.match("^MoleculeNames$", OptionsInfo["Mode"], re.I) else False

    OptionsInfo["MissingChargeMode"] = OpenFEUtil.ProcessOptionOpenFEMissingChargeMode(
        "--missingChargeMode", Options["--missingChargeMode"]
    )

    ProcessMoleculeNamesOption()

    OptionsInfo["ResultFileParams"] = OpenFEUtil.ProcessOptionOpenFEResultFileParameters(
        "--resultFileParams", Options["--resultFileParams"]
    )
    OptionsInfo["SolventParams"] = OpenFEUtil.ProcessOptionOpenFESolventParameters(
        "--solventParams", Options["--solventParams"]
    )

    ProcessOutfilePrefixOption()
    ProcessOutfileDirOption()

    OptionsInfo["Overwrite"] = Options["--overwrite"]

    # Track top level working directory...
    OptionsInfo["TopWorkingDir"] = os.getcwd()


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
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd")

    MiscUtil.ValidateOptionDirPath("-o, --outfileDir", Options["--outfileDir"])
    MiscUtil.ValidateOptionsOutputDirOverwrite(
        "-o, --outfileDir", Options["--outfileDir"], "--overwrite", Options["--overwrite"]
    )

    MiscUtil.ValidateOptionTextValue("--loggingLevel", Options["--loggingLevel"], "Info Warning Error")

    MiscUtil.ValidateOptionTextValue("-m, --mode", Options["--mode"], "FirstMolecule AllMolecules MoleculeNames")
    MiscUtil.ValidateOptionTextValue("--missingChargeMode", Options["--missingChargeMode"], "Calculate Stop")

    if re.match("^MoleculeNames$", Options["--mode"], re.I):
        if MiscUtil.IsEmpty(Options["--moleculeNames"]):
            MiscUtil.PrintError(
                'You must specify a value for "--moleculeNames" option during, MoleculeNames, value for "-m, --mode" option.'
            )


# Setup a usage string for docopt...
_docoptUsage_ = """
OpenFECalculateAbsoluteHydrationFreeEnergy.py - Calculate absolute hydration free energy

Usage:
    OpenFECalculateAbsoluteHydrationFreeEnergy.py [--ahfeParams <Name,Value,...>] [--executeDAGParams <Name,Value,..>]
                                                  [--infileParams <Name,Value,...>] [--loggingLevel <Info, Warning or Error>] [--mode <FirstMolecule, AllMolecules, or ...>]
                                                  [--missingChargeMode <Calculate or Stop>] [--moleculeNames <MolName1,MolName2,..>] [--outfilePrefix <text>]
                                                  [--resultFileParams <Name,Value,..>] [--solventParams <Name,Value,...>] [--overwrite]
                                                  [-w <dir>] -i <infile>  -o <outifiledir>
    OpenFECalculateAbsoluteHydrationFreeEnergy.py -l | --list
    OpenFECalculateAbsoluteHydrationFreeEnergy.py -h | --help | -e | --examples

Description:
    Calculate Absolute Hydration Free Energy (AHFE) for molecules in a small
    molecule input file. You may calculate AHFEs for specific molecules or all
    molecules in the input file.

    The small molecule input file must contain molecules already prepared for
    simulation. It must contain appropriate 3D coordinates along with no missing
    hydrogens.

    The MD simulation workflow, employed for the calculation of AHFEs, involves
    the following steps: initial minimization; NVT equilibration; NPT equilibration;
    production NPT. The MD simulation protocol is repeated 3 times for solvent
    transformation from MolSolventToSolventOnly and vacumm simulation, and
    he results are analyzed to estimate AHFEs. The default time and step size
    settings for the MD protocol are shown below:
        
        Protocol repeats, 3
        
        Time step size: 4.0 femtosecond

        Solvent equilibration phase:
        
        Max minimization steps: 5,000
        NVT equilibration length: 0.1 nanosecond
        NPT equilibration length: 0.2 nanosecond
        NPT length: 0.5 nanosecond
        
        Solvent production phase:
        
        Max minimization steps: 5,000
        NPT equilibration length: 1.0 nanosecond
        NPT length: 10.0 nanosecond
        
        Vacuum equilibration phase:
        
        Max minimization steps: 5,000
        NVT equilibration length: None
        NPT equilibration length: 0.2 nanosecond
        NPT length: 0.5 nanosecond
        
        Vacuum production phase:
        
        Max minimization steps: 5,000
        NVT equilibration length: None
        NPT equilibration length: 0.5 nanosecond
        NPT length: 2.0 nanosecond
        
    Each solvent and vacuum simulation, by default, may run for 11.8 and 3.2
    nanosecond respectively, for a total of 15 nanoseconds. The total MD
    simulation time for correspond to 45 nanosecond to repeat the protocol
    3 times for the solvent and vacuum simulations.

    Possible outfile prefix:
        
        <OutfilePrefix> or <InfileRoot>
        
    Possible output directories:
        
        <OutfileDir>
        
        <OutfileDir>/Transformations
        <OutfileDir>/Results
        
    Possible output files and directories under <OutfileDir>:
        
        <OutfilePrefix>_AHFE_Results.<csv or tsv>
        
        Transformations/<MolName>_Solvent_To_Solvent_Only.json
        ... ... ...
        
        Results/<MolName>_AbsoluteSolvation_Results.json
        
        Results/shared_AbsoluteSolvationSolventUnit-*/
        Results/scratch_AbsoluteSolvationVacuumUnit-*/
        ... ... ...

Options:
    -a, --ahfeParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for AHFE protocol
        settings employed during the calculation of AHFEs.
        
        The default values are automatically updated to match settings provided by
        OpenFE module AbsoluteSolvationProtocol.
        
        You must specify valid OpenFE values for these parameters. An extensive
        validation is not performed.
        
        The supported parameter names along with their default values are
        are shown below:
            
            protocolRepeats, 3
            
            Integrator settings:
            
            integratorBarostatFrequency, 25.0 * timestep  [ The specified value
                is a multiple of integratorTimestep. ]
            integratorConstraintTolerance, 1e-06
            integratorLangevinCollisionRate, 1.0  [ Units: 1 / picosecond ]
            integratorNRestartAttempts, 20
            integratorReassignVelocities, no  [ Possible values: yes or no ]
            integratorRemoveCom, no  [ Possible values: yes or no ]
            integratorTimestep, 4.0 [ Units: femtosecond ] 
            
            Lambda settings:
            
            lambdaElec, 0.0 0.25 0.5 0.75 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
                1.0  [ Possible values: A space delimited list of values
                between 0.0 and 1.0 ]
            lambdaRestraints, 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                0.0 0.0  [ Possible values: A space delimited list of values
                between 0.0 and 1.0 ]
            lambdaVdw, [0.0 0.0 0.0 0.0 0.0 0.12 0.24 0.36 0.48 0.6 0.7 0.77
                0.85 1.0  [ Possible values: A space delimited list of values
                between 0.0 and 1.0 ]
            
            Partial charge settings:
            
            partialChargeNaglModel, None  [ Default: Production AM1BCC model for
                NAGL; Possible value: Any valid name. ]
            partialChargeNumberOfConformers, None  [ Possible value: > 0 ]
            partialChargeOffToolkitBackend, AmberTools  [ Possible values:
                AmberTools or RDKit ]
            partialChargeMethod, AM1BCC  [ Possble values: AM1BCC, Espaloma,
                or NAGL ]
            
            Solvation settings:
            
            solvationBoxShape, dodecahedron  [  Possible values: cube,
                dodecahedron, or octahedron ]
            solvationBoxSize, None  [ Possible value: A triplet of space
                X Y Z values; Units: nanometer ]
            solvationSolventModel, tip3p  [ Possible values: tip3p, spce, tip4pew,
                or tip5p ]
            solvationSolventPadding, 1.5  [ Units: nanometer ]
            
            Solvent engine settings:
            
            solventEngineComputePlatform, CPU  [ Possible values: CPU, CUDA,
                OpenCL, or Reference ]
            solventEngineGpuDeviceIndex, None [ Possible values: 0, 0 1, etc. ]
            
            Solvent equil output settings:
            
            solventEquilOutputCheckpointInterval, 1.0  [ Units: nanosecond ]
            solventEquilOutputCheckpointStorageFilename, checkpoint.chk
            solventEquilOutputNPTStructure, equil_npt_structure.pdb
            solventEquilOutputNVTStructure, equil_nvt_structure.pdb
            solventEquilOutputForcefieldCache, db.json
            solventEquilOutputLogOutput, equil_simulation.log
            solventEquilOutputMinimizedStructure, minimized.pdb
            solventEquilOutputIndices, not water  [ Possible value: Any valid
                selection. ]
            solventEquilOutputPreminimizedStructure, system.pdb
            solventEquilOutputProductionTrajectoryFilename, production_equil.xtc
            solventEquilOutputTrajectoryWriteInterval, 20.0  [ Units: picosecond ]
            
            Solvent equil simulation settings:
            
            solventEquilSimulationEquilLength, 0.2  [ Units: nanosecond ]
            solventEquilSimulationEquiLengthNVT, 0.1  [ Units: nanosecond ]
            solventEquilSimulationMinimizationSteps,5000
            solventEquilSimulationProductionLength,0.5  [ Units: nanosecond ]
            
            Solvent forcefield settings:
            
            solventForcefieldConstraints, HBonds  [ Possible values: HBonds,
                AllBonds, or HAngles ]
            solventForcefields, amber/ff14SB.xml, amber/tip3p_standard.xml
                amber/tip3p_HFE_multivalent.xml amber/phosaa10.xml
                [ Possible values: A space delimited list of valid names. ]
            solventForcefieldHydrogenMass, 3.0  [ Units: amu ]
            solventForcefieldNonbondedCutoff,0.9   [ Units: nanometer ]
            solventForcefieldNonbondedMethod, PME  [ Possible values: PME or
                NoCutoff ]
            solventForcefieldRigidWater, yes,  [ Possible values: yes or no ]
            solventForcefieldSmallMoleculeForcefield, openff-2.1.1  [ Possible
                value: A valid forcefield name. ]
            
            Solvent output settings:
            
            solventOutputCheckpointInterval, 1.0  [ Units: nanosecond ]
            solventOutputCheckpointStorageFilename, solvent_checkpoint.nc
            solventOutputForcefieldCache, db.json
            solventOutputFilename, solvent.nc
            solventOutputIndices, not water   [ Possible value: Any valid
                selection. ]
            solventOutputStructure, hybrid_system.pdb
            solventOutputPositionsWriteFrequency, 100.0 [ Units: picosecond ]
            solventOutputVelocitiesWriteFrequency, None  [ Possible
                values: > 0; Units: picosecond ]
            
            Solvent simulation settings:
            
            solventSimulationEarlyTerminationTargetError, 0.0  [ Units:
                kilocalorie_per_mole ]
            solventSimulationEquilibrationLength, 1.0  [ Units: nanosecond ]
            solventSimulationMinimizationSteps, 5000
            solventSimulationNReplicas, 14
            solventSimulationProductionLength, 10.0  [ Units: nanosecond ]
            solventSimulationRealTimeAnalysisInterval, 250.0  [ Units:
                picosecond ]
            solventSimulationRealTimeAnalysisMinimumTime, 500.0 [ Units:
                picosecond
            solventSimulationSamplerMethod, repex  [ Possible values: repex,
                sams, or independent ]
            solventSimulationSamsFlatnessCriteria, logZ-flatness  [ Possible
               values: logZ-flatness, minimum-visits or histogram-flatness ]
            solventSimulationsamsGamma0, 1.0
            solventSimulationTimePerIteration,2.5  [ Units: picosecond ]
            
            Thermo settings:
            
            thermoPh, None  [ Possible values: > 0 ]
            thermoPressure, 1.0  [ Units: bar ]
            thermoRedoxPotential, None  [ Possible values: A valid float.
                Units: millivolts (mV) ]
            thermoTemperature, 298.15  [ Units: kelvin ]
            
            Vacuum engine settings:
            
            vacuumEngineComputePlatform, CPU  [ Possible values: CPU, CUDA,
                OpenCL, or Reference ]
            vacummEngineGpuDeviceIndex, None [ Possible values: 0, 0 1, etc. ]
            
            Vacuum equil output settings:
            
            vacuumEquilOutputCheckpointInterval, 1.0  [ Units: nanosecond ]
            vacuumEquilOutputCheckpointStorageFilename, checkpoint.chk
            vacuumEquilOutputNPTStructure, equil_structure.pdb
            vacuumEquilOutputNVTStructure,None
            vacuumEquilOutputForcefieldCache, db.json
            vacuumEquilOutputLogOutput, equil_simulation.log
            vacuumEquilOutputMinimizedStructure, minimized.pdb
            vacuumEquilOutputIndices, not water   [ Possible value: Any valid
                selection. ]
            vacuumEquilOutputPreminimizedStructure, system.pdb
            vacuumEquilOutputProductionTrajectoryFilename, production_equil.xtc
            vacuumEquilOutputTrajectoryWriteInterval, 20.0  [ Units: picosecond ]
            
            Vacuum equil simulation settings:
            
            vacuumEquilSimulationEquilLength, 0.2  [ Units: nanosecond ]
            vacuumEquilSimulationEquilLengthNVT, None  [ Units: nanosecond ]
            vacuumEquilSimulationMinimizationSteps, 5000
            vacuumEquilSimulationProductionLength, 0.5 [ Units: nanosecond ]
            
            Vacuum forcefield settings:
             
            vacuumForcefieldConstraints, HBonds  [ Possible values: HBonds,
                AllBonds, or HAngles ]
            vacuumForcefields, amber/ff14SB.xml, amber/tip3p_standard.xml
                amber/tip3p_HFE_multivalent.xml amber/phosaa10.xml
                [ Possible values: A space delimited list of valid names. ]
            vacuumForcefieldHydrogenMass, 3.0  [ Units: amu ]
            vacuumForcefieldNonbondedCutoff, 0.9  [ Units: nanometer ]
            vacuumForcefieldNonbondedMethod, nocutoff  [ Possible values: PME
                or NoCutoff ]
            vacuumForcefieldRigidWater, yes,  [ Possible values: yes or no ]
            vacuumForcefieldSmallMoleculeForcefield, openff-2.1.1  [ Possible
                value: A valid forcefield name. ]
            
            Vacuum output settings:
             
            vacuumOutputCheckpointInterval, 1.0  [ Units: nanosecond ]
            vacuumOutputCheckpointStorageFilename, vacuum_checkpoint.nc
            vacuumOutputForcefieldCache, db.json
            vacuumOutputFilename, vacuum.nc
            vacuumOutputIndices, not water   [ Possible value: Any valid
                selection. ]
            vacuumOutputStructure, hybrid_system.pdb
            vacuumOutputPositionsWriteFrequency, 100.0  [ Units: picosecond ]
            vacuumOutputVelocitiesWriteFrequency, None  [ Possible
                values: > 0; Units: picosecond ]
            
            Vacuum simulation settings:
             
            vacuumSimulationEarlyTerminationTargetError, 0.0  [ Units:
                0.0 kilocalorie_per_mole ]
            vacuumSimulationEquilibrationLength, 0.5  [ Units: nanosecond ]
            vacuumSimulationMinimizationSteps, 5000
            vacuumSimulationNReplicas, 14
            vacuumSimulationProductionLength, 2.0  [ Units: nanosecond ]
            vacuumSimulationRealTimeAnalysisInterval, 250.0  [ Units: picosecond ]
            vacuumSimulationRealTimeAnalysisMinimumTime, 500.0  [ Units: picosecond]
            vacuumSimulationSamplerMethod, repex [ Possible values: repex,
                sams, or independent ]
            vacuumSimulationSamsFlatnessCriteria, logZ-flatness  [ Possible
               values: logZ-flatness, minimum-visits or histogram-flatness ]
            vacuumSimulationSamsGamma0, 1.0
            vacuumSimulationTimePerIteration,2.5 [ Units: picosecond ]
            
        A brief description of parameters, taken from OpenFE documentation, is
        provided below:
            
            protocolRepeats: Number of completely independent repeats of the
                entire sampling process.
            
            Integrator settings:
            
            Parameters controlling the LangevinSplittingDynamicsMove integrator
            used for simulation.
            
            integratorBarostatFrequency: Frequency at which volume scaling
                changes should be attempted.
            integratorConstraintTolerance: Tolerance for constraint solver.
            integratorLangevinCollisionRate: Collision frequency.
            integratorNRestartAttempts: Number of attempts to restart from
                Context in case there are NaNs in the energies after
                integration.
            integratorReassignVelocities: Reassign velocities  from the
                Maxwell-Boltzmann distribution at the beginning of each
                Monte Carlo move.
            integratorRemoveCom: Remove the center of mass motion.
            integratorTimestep: Size of the simulation timestep.
            
            Lambda settings:
            
            Lambda protocol parameters.
            
            lambdaElec: List of lambda values for electrostatics. The values of
                0 and 1 imply state A and state B respectively.
            lambdaRestraints: List of lambda values for restraints. The values
                of 0 and 1 imply state A and state B respectively.
            lambdaVdw: List of lamda values for van der Waals. The values of
                of 0 and 1 imply state A and state B respectively.
            
            Partial charge settings:
            
            Parameters for automatically assigning missing partial charges to
            small molecules, including the partial charge method.
            
            partialChargeNaglModel: Model to use for partial charge assignment.
                A value of None implies the use of the latest available
                production AM1BCC model.
            partialChargeNumberOfConformers: Number of conformers to generate
                as part of the partial charge assignment. A value of None
                implies the use of the existing conformer.
            partialChargeOffToolkitBackend: OpenFF toolkit registry backend to
                use for calculating partial charges.
            partialChargeMethod: Method to use for calculating partial charges.
            
            Solvation settings:
            
            Solvation parameters for the system, including the solvent model and
            the solvent padding.
            
            solvationBoxShape: Shape of the periodic solvent box to create.
            solvationBoxSize: Lengths of the unit cell for a solvent box.
            solvationSolventModel: Forcefield water model to use during
                solvation and defining the model properties.
            solvationSolventPadding: Minimum distance from any solute bounding
                sphere to the edge of the box.
            
            Solvent engine settings:
            
            Parameters configuring the compute platform used by the OpenMM to
            perform the simulation.
            
            solventEngineComputePlatform: Platform to use for running OpenMM MD
                calculations.
            solventEngineGpuDeviceIndex: Space delimited list of device indices
                to use for running OpenMM MD calculations.
            
            Solvent equil output settings:
            
            Parameters controlling simulation output during equilibration
            phase of solvent transformation.
            
            solventEquilOutputCheckpointInterval: Frequency to write the
                checkpoint file.
            solventEquilOutputCheckpointStorageFilename: Checkpoint filename.
            solventEquilOutputNPTStructure: NPT structure filename.
            solventEquilOutputNVTStructure: NVT strucure filename.
            solventEquilOutputForcefieldCache: Filename for caching small
                molecule residue templates.
            solventEquilOutputLogOutput: Simulation log filename.
            solventEquilOutputMinimizedStructure: Minimized structire filename.
            solventEquilOutputIndices: Selection string for selecting
                coordinates to write.
            solventEquilOutputPreminimizedStructure: Initial structure filename.
            solventEquilOutputProductionTrajectoryFilename: Trajectory filename.
            solventEquilOutputTrajectoryWriteInterval: Frequency for writing
                velocities to trajectory file.
            
            Solvent equil simulation settings:
            
            Parameters controlling simulation during equilibration phase of
            solvent transformation.
            
            solventEquilSimulationEquilLength: Length of the NPT equilibration
                phase.
            solventEquilSimulationEquiLengthNVT: Length of the NVT equilibration
                phase.
            solventEquilSimulationMinimizationSteps: Maximum number of
                minimization steps to perform.
            solventEquilSimulationProductionLength: Length of the NPT production
                phase.
            
            Solvent forcefield settings:
            
            Parameters to set up the force field with OpenMM Force Fields
            equilibration phase of solvent transformation.
            
            solventForcefieldConstraints:  Constraints to use.
            solventForcefields:  List of valid forcefield paths for all
                components except small molecules.
            solventForcefieldHydrogenMass: Mass to be repartitioned to
                hydrogens from neighboring heavy atoms.
            solventForcefieldNonbondedCutoff: Cutoff for short range nonbonded
                interactions.
            solventForcefieldNonbondedMethod: Method for treating nonbonded
                interactions.
            solventForcefieldRigidWater: Use a rigid water model.
            solventForcefieldSmallMoleculeForcefield: A valid forcefield name
                to use for small molecules.
            
            Solvent output settings:
            
            Parameters controlling simulation output during final phase of
            solvent transformation.
            
            solventOutputCheckpointInterval: Frequency to write the checkpoint
                file.
            solventOutputCheckpointStorageFilename: Checkpoint filename.
            solventOutputForcefieldCache: Filename for caching small molecule
                residue templates.
            solventOutputFilename: Trajectory filename.
            solventOutputIndices: Selection string for selecting coordinates to
                write.
            solventOutputStructure: Hybrid topology structure filename.
            solventOutputPositionsWriteFrequency: Frequency for writing
                positions to trajectory file.
            solventOutputVelocitiesWriteFrequency: Frequency for writing
                velocities to trajectory file.
            
            Solvent simulation settings:
            
            Parameters controlling simulation during final phase of solvent
            transformation.
            
            solventSimulationEarlyTerminationTargetError: Target error for the
                real time analysis measured in kcal/mol. Once the MBAR error of
                the free energy is at or below this value, the simulation will
                be considered complete. The suggested value of 0.12 has shown to
                be effective in both hydration and binding free energy
                benchmarks.
            solventSimulationEquilibrationLength: Length of the equilibration
                phase. The specified value must be divisible by 'integratorTimestep'.
            solventSimulationMinimizationSteps: Maximum number of minimization
                steps to perform.
            solventSimulationNReplicas: Number of replicas to use.
            solventSimulationProductionLength: Length of the production phase.
                The specified value must be divisible by 'integratorTimestep'.
            solventSimulationRealTimeAnalysisMinimumTime: Time interval for
                performing analysis of the free energies. At each interval, real
                time analysis data will be written to a yaml file named
                <outputFileName>_real_time_analysis.yaml. The current error
                in the estimate will also be assessed and the simulation will
                be terminated when it drops below
                'solventSimulationEarlyTerminationTargetError'.
            solventSimulationSamplerMethod: Alchemical sampling method to use:
                REPEX (Hamiltonian REPlica EXchange), SAMS (Self-Adjusted
                Mixture Sampling), or Independent (Independently sampled lambda
                windows).
            solventSimulationSamsFlatnessCriteria:Method for assessing when to
                switch to asymptomatically optimal scheme for SAMS.
            solventSimulationsamsGamma0: Initial weight adaptation rate for
                SAMS.
            solventSimulationTimePerIteration: Simulation time between each
               MCMC move attempt 
            
            Vacuum engine settings:
            
            Parameters configuring the compute platform used by the OpenMM to
            perform the simulation.
            
            vacuumEngineComputePlatform: Platform to use for running OpenMM MD
                calculations.
            vacuumEngineGpuDeviceIndex: Space delimited list of device indices
                to use for running OpenMM MD calculations.
            
            The rest of the vacuum settings are similar to the solvent settings
            already described under various sections for solvent. The prefix
            'vacuum' is used for the names of the pramaters instead of the
            prefix 'solvent.'
            
            Thermo settings:
            
            Thermodynamic parameters, including the temperature and the pressure
            of the system.
            
            thermoPh: Simulation pH
            thermoPressure: Simulation pressure.
            thermoRedoxPotential:Simulation redox potential.
            thermoTemperature: Simulation temperature. 
            
    -e, --examples
        Print examples.
    --executeDAGParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for executing
        protocol DAGs (Directed Acyclic Graph) to run AHFE calculations.
        
        The supported parameter names along with their default values are
        are shown below:
            
            keepShared, yes  [ Possible values: yes or no ]
            keepScratch, no  [ Possible values: yes or no ]
            nRetries, 2  [ Possible values: >= 0. A value of 0 implies only
                1 try. ]
            
        A brief description of parameters is provided below:
            
            keepShared: Keep shared directories after the execution of DAG.
            keepScratch: Keep scratch directories after the execution of DAG.
            nRetries: Number of times to attempt the execution.
            
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file containing small molecules.
    --infileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for reading
        molecules from files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: removeHydrogens,no,sanitize,yes,strictParsing,yes
            
    -l, --list
        List default AHFE protocol settings provided by OpenFE module
        AbsoluteSolvationProtocol.
    --loggingLevel <Info, Warning or Error>  [default: Error]
        Logging level to configure the 'root logger' via logging.basicConfig()
        function. The default logging level is changed from 'logging.INFO' to
        'logging.ERROR'. Otherwise, OpenFE and its associated modules
        may generate a lot of informational messages.
    -m, --mode <FirstMolecule, AllMolecules, or ...>  [default: FirstMolecule]
        Calculate AHFE for the first molecule, the specified molecule names,
        or all molecules in an input file. Possible values:  FirstMolecule,
        AllMolecules, or MoleculeNames. You must specify a comma delimited list
        of molecule names using '--moleculeNames'option during 'MoleculeNames'
        value for '--mode' option. 
    --missingChargeMode <Calculate or Stop>  [default: Stop]
        Calculate missing partial charges for molecules before running AHFE
        calculations or terminate the execution of the script. The missing
        partial charges will be automatically calculated by OpenFE module
        AbsoluteSolvationProtocol during the calculation of AHFE. You
        may control the calculation of partial charges by specifying values for
        partialCharge* parameters using '--ahfeParams' option.
    --moleculeNames <MolName1,MolName2,..>
        A comma delimited list of molecule names for calculating AHFEs.
        This option is only used during 'MoleculeNames' value for
        '--mode' option. 
    -o, --outfileDir <outfiledir>
        Output directory.
    --outfilePrefix <text>  [default: auto]
        Prefix for generating output files under output directory.
    --overwrite
        Overwrite existing files.
    --resultFileParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        calculated RHFEs values to a results file.
        
        The supported parameter names along with their default values are
        are shown below:
            
            precision, 4  [ Possible values: > 0 ]
            delimiter, comma  [ Possible values: comma or tab ]
            
    --solventParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for solvent
        component. You must specify valid OpenFE values. No extensive validation
        is performed. These parameters are used in conjunction with solvation*
        parameters available through '--ahfeParams' to perform solvation.
        
        The supported parameter names along with their default values are
        are shown below:
            
            positiveIon, Na+ [ Possible value: Li+, Na+, K+, Rb+, or Cs+ ]
            negativeIon, Cl- [ Possible values: Cl-, Br-, F-, or I- ]
            neutralize, yes  [ Possible values: yes or no ]
            ionConcentration, 0.15  [ Units: molar ]
            
        A brief description of parameters is provided below:
            
            positiveIon, negativeion: Pair of ions used to neutralize and bring
                the solvent to required ionic concentration.
            neutralize: Neutralize the net charge on the chemical state by the
                ions in the solvent component.
            ionConcentration: Ionic concentration.
            
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    The sample protein and ligand files for tyrosine kinase 2 (Tyk2) are
    distributed with MayaChemTools and are available in data directory. These
    files have been taken from OpenFE distribution for example notebooks. The
    AM1BCC partial charges have been calculated for the ligands in SD file to
    facilitate calculations. You may review OpenFE tutorial notebooks for the
    expected results.

    To calcuate AHFE for the first molecule in a SD file, performing 3 independent
    repeats of the entire MD sampling process to estimate AHFE for the molecule,
    each solvent and vacuum MD repeat consisting of equilibration phase ( Solvent:
    Minimization - 5,000; NVT - 0.1 ns; NPT - 0.2; NPT prod - 0.5 ns; Vacuum:
    Minimization  - 5,000; NVT - None; NPT - 0.2 ns; NPT prod - 0.5 ns) and
    production phase ( Solvent: Minimization - 5,000; NPT equil - 1.0 ns; NPT prod:
    10.0 ns; Vacuum: Minimization - 5,000; NVT equil - None; NPT equil - 0.5 ns;
    NPT prod - 2 ns) using a step size of of 4 fs, writing out appropriate trajectory
    and PDB files for each MD repeat in Results subdirectory under output directory,
    type:

        % OpenFECalculateAbsoluteHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsAHFE

    To run the first example for calculating AHFE for specific molecules using CUDA
    platform on your machine to perform solvent and vacuum MD simulations and
    generate various output files, type:

        % OpenFECalculateAbsoluteHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsAHFE -m MoleculeNames
          --moleculeNames "lig_ejm_31, lig_ejm_47"
          --ahfeParams "solventEngineComputePlatform,CUDA,
          vacuumEngineComputePlatform,CUDA"

    To run the second example to see all warning messages produced by OpenFE
    modules and write various output files, type;

        % OpenFECalculateAbsoluteHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsAHFE -m MoleculeNames
          --moleculeNames "lig_ejm_31, lig_ejm_47"
          --ahfeParams "solventEngineComputePlatform,CUDA,
          vacuumEngineComputePlatform,CUDA"
          --loggingLevel Warning

    To run the first example for calculating AHFE for all molecules using CUDA
    platform on your machine to perform solvent and vacuum MD simulations,
    automatically calculate missing partial charges for molecules, and generate
    various output files, type:

        % OpenFECalculateAbsoluteHydrationFreeEnergy.py
          -i SampleTyk2LigandsNoCharges.sdf -o SampleTyk2LigandsAHFE
          -m AllMolecules --ahfeParams "solventEngineComputePlatform,CUDA,
          vacuumEngineComputePlatform,CUDA"
          --missingChargeMode Calculate

    To run the second example by specifying explict values for various parametres
    and generate various output files, type:

        % OpenFECalculateAbsoluteHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsAHFE -m MoleculeNames
          --moleculeNames "lig_ejm_31, lig_ejm_47"
          --loggingLevel Error
          --executeDAGParams "keepShared, yes, nRetries, 2"
          --missingChargeMode Stop --ahfeParams "protocolRepeats,3,
          solventEngineComputePlatform,CUDA, vacuumEngineComputePlatform,CUDA,
          integratorTimestep, 4.0, solvationBoxShape, cube,
          solvationSolventModel, tip3p,
          solventEquilSimulationEquiLengthNVT, 0.1,
          solventEquilSimulationEquilLength, 0.2,
          solventEquilSimulationProductionLength,0.5,
          solventSimulationEquilibrationLength, 1.0,
          solventSimulationProductionLength, 10.0,
          vacuumEquilSimulationEquilLengthNVT, None,
          vacuumEquilSimulationEquilLength, 0.2,
          vacuumEquilSimulationProductionLength, 0.5,
          vacuumSimulationEquilibrationLength, 0.5,
          vacuumSimulationProductionLength, 2.0,
          thermoPressure, 0.98692327, thermoTemperature, 298.15"

Author:
    Manish Sud(msud@san.rr.com)

See also:
    OpenFECalculateAbsoluteBindingFreeEnergy.py,
    OpenFECalculatePartialCharges.py, OpenFECalculateRelativeBindingFreeEnergy.py,
    OpenFECalculateRelativeHydrationFreeEnergy.py, OpenFEGenerateLigandNetwork.py

Copyright:
    Copyright (C) 2026 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using OpenFE, an
    open source molecuar for alchemical free energy calculations.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
