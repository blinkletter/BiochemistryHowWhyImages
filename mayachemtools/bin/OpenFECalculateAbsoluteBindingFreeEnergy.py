#!/usr/bin/env python
#
# File: OpenFECalculateAbsoluteBindingFreeEnergy.py
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
    from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
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
        CalculateAbsoluteBindingFreeEnergy()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def CalculateAbsoluteBindingFreeEnergy():
    """Calculate absolute binding free energy."""

    # Process input files...
    MacroMol, InMols = ProcessInputFiles()

    # Process molecule names...
    Mols = ProcessMoleculeNames(InMols)

    # Check for missing partial charges...
    CheckMissingPartialCharges(Mols)

    # Initialize ABFE protocols...
    ABFEProtocol = InitializeAbsoluteBindingProtocol()

    # Initialize solvent...
    Solvent = InitializeSolventComponent()

    # Setup transformations...
    MolTransformations = SetupTransformations(MacroMol, Mols, Solvent, ABFEProtocol)

    # Setup protocol DAGs...
    MolProtocolDAGs = SetupProtocolDAGs(MolTransformations)

    # Execute protocol DAGs and gather results...
    MolProtocolResults = ExecuteProtocolDAGsAndGatherResults(MolTransformations, MolProtocolDAGs)

    # Process protocol results...
    ProcessProtocolResults(MolTransformations, MolProtocolResults)


def InitializeAbsoluteBindingProtocol():
    """Initialize absolute solvation protocol."""

    MiscUtil.PrintInfo("\nInitializing absolute solvation protocol...")

    ABFESettings = OpenFEUtil.SetupAbsoluteBindingFreeEnergySettings("-a, --abfeParams", OptionsInfo["ABFEParams"])
    ABFEProtocol = OpenFEUtil.InitializeAbsoluteBindingFreeEngeryProtocol(ABFESettings)

    return ABFEProtocol


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


def SetupTransformations(MacroMol, Mols, Solvent, ABFEProtocol):
    """Set up transformations for molecules."""

    MiscUtil.PrintInfo("\nSetting up transformations (Count: %s)..." % (len(Mols)))

    MolTransformations = []

    for Mol in Mols:
        # Setup a chemical system for a molecule fully interacting with protein in the solvent...
        ComplexSolventSystem = OpenFEUtil.InitializeChemicalSystem(
            SmallMol=Mol, MacroMol=MacroMol, Solvent=Solvent, Name="%s_Complex_Solvent" % Mol.name
        )

        # Setup a system for a molecule fully decoupled in the solvent: Only need to use the protein and solvent....
        ProteinSolventSystem = OpenFEUtil.InitializeChemicalSystem(
            SmallMol=None, MacroMol=MacroMol, Solvent=Solvent, Name="Protein_Solvent"
        )

        # Setup a transformation for absolute binding protocol from ComplexSolventSystem to ProteinSolventSystem.
        # The AbsoluteBindingProtocol automatically creates the separate solvents state based on the complex states.
        TransformationName = "%s_AbsoluteBinding" % (Mol.name)
        MolTransformation = OpenFEUtil.InitializeTransformation(
            StateA=ComplexSolventSystem,
            StateB=ProteinSolventSystem,
            Mapping=None,
            Protocol=ABFEProtocol,
            Name=TransformationName,
            Validate=False,
        )

        MolTransformations.append(MolTransformation)

    # Write out transformations...
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

    ResultFile = "%s_ABFE_Results.%s" % (OptionsInfo["OutfilePrefix"], ResultFileParams["Ext"])
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
            DeltaGBinding = "NA"
            DeltaGBindingUncertainty = "NA"
        else:
            # Setup binding value without the units...
            DeltaGBinding = MolProtocolResult.get_estimate()
            DeltaGBinding = "%.*f" % (Precision, DeltaGBinding.m)

            # Setup uncertainty value without the units...
            DeltaGBindingUncertainty = MolProtocolResult.get_uncertainty()
            DeltaGBindingUncertainty = "%.*f" % (Precision, DeltaGBindingUncertainty.m)

        ResultData.append([MolName, DeltaGBinding, DeltaGBindingUncertainty])

    ResultDF = pd.DataFrame(ResultData, columns=["MolName", "ABFE DeltaG (kcal/mol)", "Uncertainty (kcal/mol)"])
    ResultDF.to_csv(ResultFilePath, sep=ResultFileParams["Delim"], lineterminator="\n", index=False)


def ProcessInputFiles():
    """Process input files."""

    # Read PDB file...
    MiscUtil.PrintInfo("\nReading PDB file %s..." % OptionsInfo["Infile"])
    MacroMol = OpenFEUtil.ReadPDBFile(OptionsInfo["InfilePath"], Name=OptionsInfo["InfileRoot"])

    # Read small molecule input file...
    MiscUtil.PrintInfo("\nReading small molecule file %s..." % OptionsInfo["SmallMolFile"])
    Mols, MolCount, ValidMolCount = OpenFEUtil.ReadAndValidateMolecules(
        OptionsInfo["SmallMolFilePath"], **OptionsInfo["SmallMolFileParams"]
    )

    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount))

    if ValidMolCount == 0:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("No valid molecules found in small molecule input file.\n")

    return (MacroMol, Mols)


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
            'The small molecule input file contains molecules with missing partial charges. The execution of the script has been terminated for "Stop" value of "--missingChargedMode" option. You may continue the execution of the script by specifying "Calculate" value for "--missingChargedMode" option.\n\nThe missing charges will be automatically calculated by OpenFE AbsoluteBindingProtocol module during the calculation of ABFE. You may control the calculation of partial charges by specifying values for partialCharge* parameters using "--abfe" option.  Alternatively, you may employ the OpenFECalculatePartialCharges.py script to calculate partial charges and use the small molecule input file containing charges to calculate ABFE.\n'
        )
    else:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintWarning(
            'The small molecule input file contains molecules with missing partial charges. The missing charges will be automatically calculated by OpenFE AbsoluteBindingProtocol module during the calculation of ABFE. You may control the calculation of partial charges by specifying values for partialCharge* parameters using "--abfeParams" option. Alternatively, you may employ the OpenFECalculatePartialCharges.py script to calculate partial charges and use the small molecule input file containing charges to calculate ABFE.\n'
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
        OutfilePrefix = OptionsInfo["SmallMolFileRoot"]

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

    ABFESettings = AbsoluteBindingProtocol.default_settings()

    MiscUtil.PrintInfo("\nListing ABFE settings...")
    OpenFEUtil.ListOpenFESettings(ABFESettings)


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

    OptionsInfo["SmallMolFile"] = Options["--smallMolFile"]
    OptionsInfo["SmallMolFilePath"] = os.path.abspath(OptionsInfo["SmallMolFile"])
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["SmallMolFile"])
    OptionsInfo["SmallMolFileRoot"] = FileName

    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["SmallMolFileParams"] = MiscUtil.ProcessOptionInfileParameters(
        "--smallMolFileParams",
        Options["--smallMolFileParams"],
        InfileName=Options["--smallMolFile"],
        ParamsDefaultInfo=ParamsDefaultInfoOverride,
    )

    ParamsDefaultInfoOverride = {"EngineComputePlatform": "CPU"}
    OptionsInfo["ABFEParams"] = OpenFEUtil.ProcessOptionOpenFEAbsoluteBindingFreeEnergyParameters(
        "-a, --abfeParams", Options["--abfeParams"], ParamsDefaultInfo=ParamsDefaultInfoOverride
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
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "pdb cif")

    MiscUtil.ValidateOptionFilePath("-s, --smallMolFile", Options["--smallMolFile"])
    MiscUtil.ValidateOptionFileExt("-s, --smallMolFile", Options["--smallMolFile"], "sdf sd")

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
OpenFECalculateAbsoluteBindingFreeEnergy.py - Calculate absolute binding free energy

Usage:
    OpenFECalculateAbsoluteBindingFreeEnergy.py [--abfeParams <Name,Value,...>] [--executeDAGParams <Name,Value,..>]
                                                  [--loggingLevel <Info, Warning or Error>] [--mode <FirstMolecule, AllMolecules, or ...>]
                                                  [--missingChargeMode <Calculate or Stop>] [--moleculeNames <MolName1,MolName2,..>]
                                                  [--outfilePrefix <text>] [--overwrite] [--resultFileParams <Name,Value,..>] [--solventParams <Name,Value,...>]
                                                  [--smallMolFileParams <Name,Value,...> ] [-w <dir>] -i <infile>   -s <smallmolfile> -o <outifiledir>
    OpenFECalculateAbsoluteBindingFreeEnergy.py -l | --list
    OpenFECalculateAbsoluteBindingFreeEnergy.py -h | --help | -e | --examples

Description:
    Calculate Absolute Binding Free Energy (ABFE) for molecules in a small
    molecule input file. You may calculate ABFEs for specific molecules or all
    molecules in the input file.

    The input file must contain a macromolecule already prepared for simulation.
    The preparation of the macromolecule for a simulation generally involves the
    following tasks: identification and replacement of non-standard residues;
    addition of missing residues; addition of missing heavy atoms; addition of
    missing hydrogens.

    In addition, the small molecule input file must contain molecules already
    prepared for simulation. It must contain appropriate 3D coordinates relative
    to the macromolecule along with no missing hydrogens.

    The MD simulation workflow, employed for the calculation of ABFEs, involves
    the following steps:
        
        Protocol repeats, 3
        
        Time step size: 4.0 femtosecond

        Complex equilibration phase:
        
        Max minimization steps: 5,000
        NVT equilibration length: 0.25 nanosecond
        NPT equilibration length: 0.5 nanosecond
        NPT length: 5.0 nanosecond
        
        Complex production phase:
        
        Max minimization steps: 5,000
        NPT equilibration length: 1.0 nanosecond
        NPT length: 10.0 nanosecond

        Solvent equilibration phase:
        
        Max minimization steps: 5,000
        NVT equilibration length: 0.1 nanosecond
        NPT equilibration length: 0.2 nanosecond
        NPT length: 0.5 nanosecond
        
        Solvent production phase:
        
        Max minimization steps: 5,000
        NPT equilibration length: 1.0 nanosecond
        NPT length: 10.0 nanosecond

    Each complex and solvent simulation, by default, may run for 16.75 and 11.8
    nanosecond respectively, for a total of 28.55 nanoseconds. The total MD
    simulation time for correspond to 85.66 nanosecond to repeat the protocol
    3 times for the complex and solvent simulations.

    Possible outfile prefix:
        
        <OutfilePrefix> or <SmallMolFileRoot>
        
    Possible output directories:
        
        <OutfileDir>
        
        <OutfileDir>/Transformations
        <OutfileDir>/Results
        
    Possible output files and directories under <OutfileDir>:
        
        <OutfilePrefix>_ABFE_Results.<csv or tsv>
        
        Transformations/<MolName>_AbsoluteBinding.json
        ... ... ...
        
        Results/<MolName>_AbsoluteBinding_Results.json
        
        Results/shared_AbsoluteBindingComplexUnit-*/
        Results/shared_AbsoluteBindingSolventUnit-*/
        ... ... ...

Options:
    -a, --abfeParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for ABFE protocol
        settings employed during the calculation of ABFEs.
        
        The default values are automatically updated to match settings provided by
        OpenFE module AbsoluteBindingProtocol.
        
        You must specify valid OpenFE values for these parameters. An extensive
        validation is not performed.
        
        The supported parameter names along with their default values are
        are shown below: explain and doc...
            
            protocolRepeats, 3
            
            Complex equil output settings:
            
            complexEquilOutputCheckpointInterval, 1  [ Units: nanosecond ]
            complexEquilOutputCheckpointStorageFilename, checkpoint.chk
            complexEquilOutputEquilNPTStructure, equil_npt_structure.pdb
            complexEquilOutputEquilNVTstructure, equil_nvt_structure.pdb
            complexEquilOutputForcefieldCache, db.json
            complexEquilOutputLogOutput, production_equil_simulation.log
            complexEquilOutputMinimizedStructure, minimized.pdb
            complexEquilOutputIndices, all  [ Possible value: Any valid
                selection. ]
            complexEquilOutputPremnimizedStructure, system.pdb
            complexEquilOutputProductionTrajectoryFilename, production_equil.xtc
            complexEquilOutputTrajectoryWriteInterval,  20.0  [ Units:
                picosecond ]
            
            Complex equil simulation settings:
            
            complexEquilSimulationEquilibrationLength, 0.5 [ Units: nanosecond ]
            complexEquilSimulationEquilibrationLengthNVT, 0.25   [ Units:
                nanosecond ]
            complexEquilSimulationMinimizationSteps, 5000
            complexEquilSimulationProductionLength, 5.0  [ Units: nanosecond ]
            
            Complex lambda settings:
            
            complexLambdaElec, 0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.2 0.3 0.4 0.5 0.6
                0.7 0.8 0.9 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
                1.0 1.0  [ Possible values: A space delimited list of values
                between 0.0 and 1.0 ]
            complexLambdaRestraints, 0.0 0.2 0.4 0.6 0.8 1.0 1.0 1.0 1.0 1.0
                1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
                1.0 1.0 1.0 1.0  [ Possible values: A space delimited list of
                values between 0.0 and 1.0 ]
            complexLambdaVdw, 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                0.0 0.0 0.0 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85
                0.9 0.95 1.0  [ Possible values: A space delimited list of values
                between 0.0 and 1.0 ]
            
            Complex output settings:
            
            complexOutputCheckpointInterval, 1.0  [ Units: nanosecond ]
            complexOutputCheckpointStorageFilename, complex_checkpoint.nc
            complexOutputForcefieldCache, db.json
            complexOutputFilename, complex.nc
            complexOutputIndices, not water   [ Possible value: Any valid
                selection. ]
            complexOutputStructure, alchemical_system.pdb
            complexOutputPositionsWriteFrequency, 100.0  [ Units: picosecond ]
            complexOutputVelocitiesWriteFrequency, None  [ Possible
                values: > 0; Units: picosecond ]
            
            Complex simulation settings:
            
            complexSimulationEarlyTerminationTargetError, 0.0  [ Units:
                kilocalorie_per_mole ]
            complexSimulationEquilibrationLength,  1.0  [ Units: nanosecond ]
            complexSimulationMinimizationSteps, 5000
            complexSimulationNReplicas, 30
            complexSimulationProductionLength, 10.0  [ Units: nanosecond ]
            complexSimulationRealTimeAnalysisInterval, 250.0  [ Units:
                picosecond ]
            complexSimulationRealTimeAnalysisMinimumTime, 500.0  [ Units:
                picosecond ]
            complexSimulationSamplerMethod, repex  [ Possible values: repex,
                sams, or independent ]
            complexSimulationSamsFlatnessCriteria, logZ-flatness  [ Possible
                values: logZ-flatness, minimum-visits or histogram-flatness ]
            complexSimulationSamsGamma0, 1.0
            complexSimulationTimePerIteration, 2.5   [ Units: picosecond ]
            
            Complex solvation settings:
            
            complexSolvationBoxShape, dodecahedron  [  Possible values: cube,
                dodecahedron, or octahedron ]
            complexSolvationBoxSize, None  [ Possible value: A triplet of space
                X Y Z values; Units: nanometer ]
            complexSolvationSolventModel, tip3p  [ Possible values: tip3p, spce,
                tip4pew, or tip5p ]
            complexSolvationSolventPadding, 1.0  [ Units: nanometer ]
            
            Engine settings:
            
            engineComputePlatform, CPU  [ Possible values: CPU, CUDA,
                OpenCL, or Reference ]
            engineGpuDeviceIndex, None [ Possible values: 0, 0 1, etc. ]
             
            Forcefield settings:
            
            forcefieldConstraints, HBonds  [ Possible values: HBonds,
                AllBonds, or HAngles ]
            forcefields, amber/ff14SB.xml amber/tip3p_standard.xml
                amber/tip3p_HFE_multivalent.xml amber/phosaa10.xml
                [ Possible values: A space delimited list of valid names. ]
            forcefieldHydrogenMass, 3.0  [ Units: amu ]
            forcefieldNonbondedCutoff, 0.9   [ Units: nanometer ]
            forcefieldNonbondedMethod, PME  [ Possible values: PME or
                NoCutoff ]
            forcefieldRigidWater, yes,  [ Possible values: yes or no ]
            forcefieldSmallMoleculeForcefield, openff-2.1.1  [ Possible
                value: A valid forcefield name. ]
            
            Integrator settings:
            
            integratorBarostatFrequency, 25.0 * timestep  [ The specified value
                is a multiple of integratorTimestep. ]
            integratorConstraintTolerance, 1e-06
            integratorLangevinCollisionRate, 1.0  [ Units: 1 / picosecond ]
            integratorNRestartAttempts, 20
            integratorReassignVelocities, no  [ Possible values: yes or no ]
            integratorRemoveCom, no  [ Possible values: yes or no ]
            integratorTimestep, 4.0 [ Units: femtosecond ] 
            
            Partial charge settings:
            
            partialChargeNaglModel, None  [ Default: Production AM1BCC model for
                NAGL; Possible value: Any valid name. ]
            partialChargeNumberOfConformers, None  [ Possible value: > 0 ]
            partialChargeOffToolkitBackend, AmberTools  [ Possible values:
                AmberTools or RDKit ]
            partialChargeMethod, AM1BCC  [ Possble values: AM1BCC, Espaloma,
                or NAGL ]
            
            Restraint settings:
            
            restraintKPhiA, 334.72  [ Units: kilojoule_per_mole / radian**2
                The default value is equivalent to 80 kcal/mol/radian**2 ]
            restraintKPhiB, 334.72  [ Units: kilojoule_per_mole / radian**2
                The default value is equivalent to 80 kcal/mol/radian**2 ]
            restraintKPhiC, 334.72  [ Units: kilojoule_per_mole / radian**2
                The default value is equivalent to 80 kcal/mol/radian**2 ]
            restraintKR, 4184.0  [ Units: kilojoule_per_mole / nanometer**2
                 The default value is equivalent to 10 kcal/mol/angstrom**2
            restraintKThetaA, 334.72  [ Units: kilojoule_per_mole / radian**2
                The default value is equivalent to 80 kcal/mol/radian**2 ]
            restraintKThetaB, 334.72  [ Units: kilojoule_per_mole / radian**2
                The default value is equivalent to 80 kcal/mol/radian**2 ]
            restraintAnchorFindingStrategy, bonded  [ Possible values:
                multi-residue or bonded ] 
            restraintDsspFilter, yes   [ Possible values: yes or no ]
            restraintHostMaxDistance, 1.5  [ Units: nanometer ]
            restraintHostMinDistance, 0.5  [ Units: nanometer ]
            restraintHostSelection, backbone   [ Possible value: Any valid
                selection. ]
            restraintRmsfCutoff, 0.1  [ Units: nanometer ]
            
            Solvent equil output settings:
            
            solventEquilOutputCheckpointInterval, 1.0  [ Units: nanosecond ]
            solventEquilOutputCheckpointStorageFilename, checkpoint.chk
            solventEquilEquilOutputNPTStructure, equil_npt_structure.pdb
            solventEquilEquilNVTOutputStructure, equil_nvt_structure.pdb
            solventEquilOutputForcefieldCache, db.json
            solventEquilOutputLogOutput, production_equil_simulation.log
            solventEquilOutputMinimizedStructure, minimized.pdb
            solventEquilOutputIndices, all  [  Possible value: Any valid
                selection. ]
            solventEquilOutputPreminimizedStructure, system.pdb
            solventEquilOutputProductionTrajectoryFilename, production_equil.xtc
            solventEquilOutputTrajectoryWriteInterval, 20.0  [ Units:
                picosecond ]
            
            Solvent_equil_simulation_settings:
            
            solventEquilSimulationEquilibrationLength, 0.2 [ Units: nanosecond ]
            solventEquilSimulationEquilibrationLengthNVT, 0.1  [ Units:
                nanosecond ]
            solventEquilSimulationMinimizationSteps, 5000
            solventEquilSimulationProductionLength, 0.5  [ Units: nanosecond ]
            
            Solvent lambda settings:
            
            solventLambdaElec, 0.0 0.25 0.5 0.75 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
                1.0 1.0  [ Possible values: A space delimited list of values
                between 0.0 and 1.0 ]
            solventLambdaRestraints, 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                0.0 0.0 0.0  [ Possible values: A space delimited list of values
                between 0.0 and 1.0 ]
            solventLambdaVdw, 0.0 0.0 0.0 0.0 0.0 0.12 0.24 0.36 0.48 0.6 0.7
                0.77 0.85 1.0  [ Possible values: A space delimited list of values
                between 0.0 and 1.0 ]
            
            Solvent output settings:
            
            solventOutputCheckpointInterval, 1.0  [ Units: nanosecond ]
            solventOutputCheckpointStorageFilename, solvent_checkpoint.nc
            solventOutputForcefieldCache, db.json
            solventOutputFilename, solvent.nc
            solventOutputIndices, not water  [ Possible value: Any valid
                selection. ]
            solventOutputStructure, alchemical_system.pdb
            solventOutputPositionsWriteFrequency, 100.0  [ Units: picosecond ]
            solventOutputVelocitiesWriteFrequency, None  [ Possible
                values: > 0; Units: picosecond ]
            
            Solvent simulation settings:
            
            solventSimulationEarlyTerminationTargetError, 0.0  [ Units:
                kilocalorie_per_mole ]
            solventSimulationEquilibrationLength, 1.0  [ Units: nanosecond ]
            solventSimulationMinimizationSteps, 5000
            solventSimulationNReplicas, 14
            solventSimulationProductionLength, 10.0  [ Units: nanosecond ]
            solventSimulationRealTimeAnalysisInterval, 250.0  [ Unit: picosecond ]
            solventSimulationRealTimeAnalysisMinimumTime, 500.0  [ Units:
                picosecond ]
            solventSimulationSamplerMethod, repex  [ Possible values: repex,
                sams, or independent ]
            solventSimulationSamsFlatnessCriteria, logZ-flatness  [ Possible
                values: logZ-flatness, minimum-visits or histogram-flatness ]
            solventSimulationSamsGamma0, 1.0
            solventSimulationTimePerIteration, 2.5  [ Units: picosecond ]
            
            Solvent solvation settings:
            
            solventSolvationBoxShape, dodecahedron  [  Possible values: cube,
                dodecahedron, or octahedron ]
            solventSolvationBoxSize, None  [ Possible value: A triplet of space
                X Y Z values; Units: nanometer ]
            solventSolvationSolventModel, tip3p  [ Possible values: tip3p, spce,
                tip4pew, or tip5p ]
            solventSolvationSolventPadding, 1.5  [ Units: nanometer ]
            
            Thermo settings:
            
            thermoPh, None  [ Possible values: > 0 ]
            thermoPressure, 1.0  [ Units: bar ]
            thermoRedoxPotential, None  [ Possible values: A valid float.
                Units: millivolts (mV) ]
            thermoTemperature, 298.15  [ Units: kelvin ]
            
        A brief description of parameters, taken from OpenFE documentation, is
        provided below:
            
            protocolRepeats: Number of completely independent repeats of the
                entire sampling process.
            
            Complex settings:
            
            Complex parameters for the system, including the solvent model and
            the solvent padding.
            
            Complex equil output settings:
            
            Parameters controlling simulation output during equilibration
            phase of complex transformation.
            
            complexEquilOutputCheckpointInterval: Frequency to write the
                checkpoint file.
            complexEquilOutputCheckpointStorageFilename: Checkpoint filename.
            complexEquilOutputEquilNPTStructure: NPT structure filename.
            complexEquilOutputEquilNVTstructure: NVT strucure filename.
            complexEquilOutputForcefieldCache:  Filename for caching small
                molecule residue templates.
            complexEquilOutputLogOutput: Simulation log filename.
            complexEquilOutputMinimizedStructure: Minimized structire filename.
            complexEquilOutputIndices: Selection string for selecting
                coordinates to write.
            complexEquilOutputPremnimizedStructure: Initial structure filename.
            complexEquilOutputProductionTrajectoryFilename: Trajectory filename.
            complexEquilOutputTrajectoryWriteInterval: Frequency for writing
                velocities to trajectory file.
            
            Complex equil simulation settings:
            
            Parameters controlling simulation during equilibration phase of
            complex transformation.
            
            complexEquilSimulationEquilibrationLength:  Length of the NPT
                equilibration phase.
            complexEquilSimulationEquilibrationLengthNVT: Length of the NVT
                equilibration phase.
            complexEquilSimulationMinimizationSteps: Maximum number of
                minimization steps to perform.
            complexEquilSimulationProductionLength:  Length of the NPT
                production phase.
            
            Complex lambda settings:
            
            Lambda protocol parameters for complex transformation.
            
            complexLambdaElec: List of lambda values for electrostatics. The
                values of 0 and 1 imply state A and state B respectively.
            complexLambdaRestraints: List of lambda values for restraints. The
                values of 0 and 1 imply state A and state B respectively.
            complexLambdaVdw: List of lamda values for van der Waals. The
                values of of 0 and 1 imply state A and state B respectively.
            
            Complex output settings:
            
            Parameters controlling simulation output during final phase of
            complex transformation.
            
            complexOutputCheckpointInterval:  Frequency to write the checkpoint
                file.
            complexOutputCheckpointStorageFilename: Checkpoint filename.
            complexOutputForcefieldCache: Filename for caching small molecule
                residue templates.
            complexOutputFilename: Trajectory filename.
            complexOutputIndices: Selection string for selecting coordinates to
                write.
            complexOutputStructure: Topology structure filename.
            complexOutputPositionsWriteFrequency: Frequency for writing
                positions to trajectory file.
            complexOutputVelocitiesWriteFrequency: Frequency for writing
                velocities to trajectory file.
            
            Complex simulation settings:
            
            Parameters controlling simulation during final phase of complex
            transformation.
            
            complexSimulationEarlyTerminationTargetError: Target error for the
                real time analysis measured in kcal/mol. Once the MBAR error of
                the free energy is at or below this value, the simulation will
                be considered complete. The suggested value of 0.12 has shown to
                be effective in both hydration and binding free energy
                benchmarks.
            complexSimulationEquilibrationLength: Length of the equilibration
                phase. The specified value must be divisible by
                'integratorTimestep'.
            complexSimulationMinimizationSteps: Maximum number of minimization
                steps to perform.
            complexSimulationNReplicas: Number of replicas to use.
            complexSimulationProductionLength: Length of the production phase.
                The specified value must be divisible by 'integratorTimestep'.
            complexSimulationRealTimeAnalysisMinimumTime: Time interval for
                performing analysis of the free energies. At each interval, real
                time analysis data will be written to a yaml file named
                <outputFileName>_real_time_analysis.yaml. The current error
                in the estimate will also be assessed and the simulation will
                be terminated when it drops below
                'complexSimulationEarlyTerminationTargetError'.
            complexSimulationSamplerMethod: Alchemical sampling method to use:
                REPEX (Hamiltonian REPlica EXchange), SAMS (Self-Adjusted
                Mixture Sampling), or Independent (Independently sampled lambda
                windows).
            complexSimulationSamsFlatnessCriteria:Method for assessing when to
                switch to asymptomatically optimal scheme for SAMS.
            complexSimulationsamsGamma0: Initial weight adaptation rate for
                SAMS.
            complexSimulationTimePerIteration: Simulation time between each
               MCMC move attempt 
            
            Complex solvation settings:
            
            Solvation parameters for the system, including the solvent model and
            the solvent padding.
            
            complexSolvationBoxShape: Shape of the periodic solvent box.
            complexSolvationBoxSize:  Lengths of the unit cell for a solvent box.
            complexSolvationSolventModel: Forcefield water model to use during
                solvation and defining the model properties.
            complexSolvationSolventPadding: Minimum distance from any solute
                bounding sphere to the edge of the box.
            
            Engine settings:
            
            Parameters configuring the compute platform used by the OpenMM to
            perform the simulation.
            
            engineComputePlatform: Platform to use for running OpenMM MD
                calculations.
            engineGpuDeviceIndex: Space delimited list of device indices
                to use for running OpenMM MD calculations.
            
            Forcefield settings:
            
            forcefieldConstraints:Constraints  to use.
            forcefields: List of valid forcefield paths for all components
                except small molecules.
            forcefieldHydrogenMass: Mass to be repartitioned to hydrogens
                from neighboring heavy atoms.
            forcefieldNonbondedCutoff: Cutoff for short range nonbonded
                interactions.
            forcefieldNonbondedMethod: Method for treating nonbonded
                interactions.
            forcefieldRigidWater: Use a rigid water model.
            forcefieldSmallMoleculeForcefield: A valid forcefield name to use
                for small molecules.
            
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
            
            Restraint settings:
            
            Parameters to configure Boresch-style restraint between two groups
            of atoms named host  (Hx) and guest (Gx).
            
            restraintKPhiA: Equilibrium force constant for the dihedral formed
                by H2-H1-H0-G0.
            restraintKPhiB: Equilibrium force constant for the dihedral formed
                by H1-H0-G0-G1.
            restraintKPhiC: Equilibrium force constant for the dihedral formed
                by H0-G0-G1-G2.
            restraintKR: Bond spring constant between H0 and G0.
            restraintKThetaA: Spring constant for the angle formed by H1-H0-G0.
            restraintKThetaB: Spring constant for the angle formed by H0-G0-G1
            restraintAnchorFindingStrategy: Boresch atom picking strategy to
                use. bonded: pick host atoms that are bonded to each other.
                multi-residue: pick host atoms which can span multiple residues.
            restraintDsspFilter: Apply DSSP filter to the host atoms.
            restraintHostMaxDistance: Minimum distance between any host atom
                and the guest G0 atom.
            restraintHostMinDistance: Xaximum distance between any host atom
                and the guest G0 atom
            restraintHostSelection: A valid selection string to sub-select the
                host atoms which will be involved in the restraint.
            restraintRmsfCutoff: Cutoff value for filtering atoms by their root
                mean square fluctuation. Atoms with values above this cutoff
                are ignored.
            
            Solvent equil output settings:
            Solvent equil simulation settings:
            Solvent lambda settings:
            Solvent output settings:
            Solvent simulation settings:
            Solvent solvation settings:
            
            The solvent settings are similar to the complex settings already
            described under various sections for complex. The prefix 'solvent'
            is used for the names of the pramaters instead of the prefix
            'complex.'
            
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
        protocol DAGs (Directed Acyclic Graph) to run ABFE calculations.
        
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
        Input file name containing a macromolecule.
    -l, --list
        List default ABFE protocol settings provided by OpenFE module
        AbsoluteBindingProtocol.
    --loggingLevel <Info, Warning or Error>  [default: Error]
        Logging level to configure the 'root logger' via logging.basicConfig()
        function. The default logging level is changed from 'logging.INFO' to
        'logging.ERROR'. Otherwise, OpenFE and its associated modules
        may generate a lot of informational messages.
    -m, --mode <FirstMolecule, AllMolecules, or ...>  [default: FirstMolecule]
        Calculate ABFE for the first molecule, the specified molecule names,
        or all molecules in an input file. Possible values:  FirstMolecule,
        AllMolecules, or MoleculeNames. You must specify a comma delimited list
        of molecule names using '--moleculeNames'option during 'MoleculeNames'
        value for '--mode' option. 
    --missingChargeMode <Calculate or Stop>  [default: Stop]
        Calculate missing partial charges for molecules before running ABFE
        calculations or terminate the execution of the script. The missing
        partial charges will be automatically calculated by OpenFE module
        AbsoluteBindingProtocol during the calculation of ABFE. You
        may control the calculation of partial charges by specifying values for
        partialCharge* parameters using '--abfe' option.
    --moleculeNames <MolName1,MolName2,..>
        A comma delimited list of molecule names for calculating ABFEs.
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
        parameters available through '--abfeParams' to perform solvation.
        
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
            
    -s, --smallMolFile <SmallMolFile>
        Input file containing small molecules.
    --smallMolFileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for reading
        molecules from files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: removeHydrogens,no,sanitize,yes,strictParsing,yes
            
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    The sample protein and ligand files for tyrosine kinase 2 (Tyk2) are
    distributed with MayaChemTools and are available in data directory. These
    files have been taken from OpenFE distribution for example notebooks. The
    AM1BCC partial charges have been calculated for the ligands in SD file to
    facilitate calculations. You may review OpenFE tutorial notebooks for the
    expected results.

    To calcuate ABFE for the first molecule in a SD file, performing 3 independent
    repeats of the entire MD sampling process to estimate ABFE for the molecule,
    each solvent and vacuum MD repeat consisting of equilibration phase ( Complex:
    Minimization - 5,000; NVT - 0.25 ns; NPT - 0.5; NPT prod - 5.0 ns; Solvent:
    Minimization  - 5,000; NVT - 0.1; NPT - 0.2 ns; NPT prod - 0.5 ns) and
    production phase ( Complex: Minimization - 5,000; NPT equil - 1.0 ns; NPT prod:
    10.0 ns; Solvent: Minimization - 5,000; NPT equil - 1.0 ns; NPT prod - 10 ns)
    using a step size of of 4 fs, writing out appropriate trajectory and PDB files for
    each MD repeat in Results subdirectory under output directory, type:

        % OpenFECalculateAbsoluteBindingFreeEnergy.py -i SampleTyk2.pdb
          -s SampleTyk2Ligands.sdf -o SampleTyk2LigandsAHFE

    To run the first example for calculating ABFE for specific molecules using CUDA
    platform on your machine to perform solvent and vacuum MD simulations and
    generate various output files, type:

        % OpenFECalculateAbsoluteBindingFreeEnergy.py -i SampleTyk2.pdb
          -s SampleTyk2Ligands.sdf -o SampleTyk2LigandsAHFE -m MoleculeNames
          --moleculeNames "lig_ejm_31, lig_ejm_47"
          --abfeParams "engineComputePlatform,CUDA"

    To run the second example to see all warning messages produced by OpenFE
    modules and write various output files, type;

        % OpenFECalculateAbsoluteBindingFreeEnergy.py -i SampleTyk2.pdb
          -s SampleTyk2Ligands.sdf -o SampleTyk2LigandsAHFE -m MoleculeNames
          --moleculeNames "lig_ejm_31, lig_ejm_47"
          --abfeParams "engineComputePlatform,CUDA"
          --loggingLevel Warning

    To run the first example for calculating ABFE for all molecules using CUDA
    platform on your machine to perform solvent and vacuum MD simulations,
    automatically calculate missing partial charges for molecules, and generate
    various output files, type:

        % OpenFECalculateAbsoluteBindingFreeEnergy.py -i SampleTyk2.pdb
          -s SampleTyk2LigandsNoCharges.sdf -o SampleTyk2LigandsAHFE
          -m AllMolecules --abfeParams "engineComputePlatform,CUDA"
          --missingChargeMode Calculate

    To run the second example by specifying explict values for various parametres
    and generate various output files, type:

        % OpenFECalculateAbsoluteBindingFreeEnergy.py -i SampleTyk2.pdb
          -s SampleTyk2Ligands.sdf -o SampleTyk2LigandsAHFE -m MoleculeNames
          --moleculeNames "lig_ejm_31, lig_ejm_47"
          --loggingLevel Error
          --executeDAGParams "keepShared, yes, nRetries, 2"
          --missingChargeMode Stop --abfeParams "protocolRepeats,3,
          engineComputePlatform,CUDA,integratorTimestep, 4.0,
          complexEquilSimulationEquilibrationLengthNVT, 0.25,
          complexEquilSimulationEquilibrationLength, 0.5,
          complexEquilSimulationProductionLength,5.0,
          complexSimulationEquilibrationLength, 1.0,
          complexSimulationProductionLength, 10.0,
          solventEquilSimulationEquilibrationLengthNVT, 0.1,
          solventEquilSimulationEquilibrationLength, 0.2,
          solventEquilSimulationProductionLength, 0.5,
          solventSimulationEquilibrationLength, 1.0,
          solventSimulationProductionLength, 10.0,
          thermoPressure, 1.0, thermoTemperature, 298.15"

Author:
    Manish Sud(msud@san.rr.com)

See also:
    OpenFECalculateAbsoluteHydrationFreeEnergy.py,
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
