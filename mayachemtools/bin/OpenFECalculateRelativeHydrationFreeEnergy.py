#!/usr/bin/env python
#
# File: OpenFECalculateRelativeHydrationFreeEnergy.py
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
import numpy as np
import pandas as pd

# OpenFE imports...
try:
    import openfe
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
        CalculateRelativeHydrationFreeEnergy()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def CalculateRelativeHydrationFreeEnergy():
    """Calculate relative hydration free energy."""

    # Process input file...
    Mols = ProcessInputFile()

    # Validate molecule names...
    ValidateMoleculeNames(Mols)

    # Check for miising partial charges...
    CheckMissingPartialCharges(Mols)

    # Setup atom mapping...
    MolAToMolBMappings = GenerateAtomMappings(Mols)

    # Initialize RHFE protocols...
    RHFEProtocol, RHFEProtocolChargeCorrection, RHFEProtocolVacuum = InitializeRelativeHybridTopologyProtocol()

    # Initialize solvent...
    Solvent = InitializeSolventComponent()

    # Setup transformations...
    MolAToMolBTransformations = SetupTransformations(
        MolAToMolBMappings, Solvent, RHFEProtocol, RHFEProtocolChargeCorrection, RHFEProtocolVacuum
    )

    # Setup protocol DAGs...
    MolAToMolBProtocolDAGs = SetupProtocolDAGs(MolAToMolBTransformations)

    # Execute protocol DAGs and gather results...
    MolAToMolBProtocolResults = ExecuteProtocolDAGsAndGatherResults(MolAToMolBTransformations, MolAToMolBProtocolDAGs)

    # Process protocol results...
    ProcessProtocolResults(MolAToMolBTransformations, MolAToMolBProtocolResults)


def InitializeRelativeHybridTopologyProtocol():
    """Initialize relative hybrid toplology protocol."""

    MiscUtil.PrintInfo("\nInitializing relative hybrid topology protocol...")

    RHFESettings = OpenFEUtil.SetupRelativeFreeEnergySettings("-r, --rhfeParams", OptionsInfo["RHFEParams"])
    RHFEProtocol = OpenFEUtil.InitializeRelativeFreeEngeryHybridTopologyProtocol(RHFESettings)

    RHFESettingsChargeCorrection = OpenFEUtil.SetupRelativeFreeEnergySettings(
        "-r, --rhfeParams", OptionsInfo["RHFEParams"]
    )
    OpenFEUtil.UpdateRelativeFreeEnergySettingsForChargeCorrection(
        "--rhfeChargeCorrectionParams", OptionsInfo["RHFEChargeCorrectionParams"], RHFESettingsChargeCorrection
    )
    RHFEProtocolChargeCorrection = OpenFEUtil.InitializeRelativeFreeEngeryHybridTopologyProtocol(
        RHFESettingsChargeCorrection
    )

    RHFESettingsVacuum = OpenFEUtil.SetupRelativeFreeEnergySettings("-r, --rhfeParams", OptionsInfo["RHFEParams"])
    OpenFEUtil.UpdateRelativeFreeEnergySettingsForVacuum(
        "--rhfeVacuumParams", OptionsInfo["RHFEVacuumParams"], RHFESettingsVacuum
    )
    RHFEProtocolVacuum = OpenFEUtil.InitializeRelativeFreeEngeryHybridTopologyProtocol(RHFESettingsVacuum)

    return (RHFEProtocol, RHFEProtocolChargeCorrection, RHFEProtocolVacuum)


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


def SetupTransformations(MolAToMolBMappings, Solvent, RHFEProtocol, RHFEProtocolChargeCorrection, RHFEProtocolVacuum):
    """Set up a transformation pair for each mapping."""

    MiscUtil.PrintInfo("\nSetting up alchemical transformations (Count: %s)..." % (len(MolAToMolBMappings) * 2))

    MolAToMolBTransformations = []

    for MolAToMolBMapping in MolAToMolBMappings:
        MolA = MolAToMolBMapping.componentA
        MolB = MolAToMolBMapping.componentB

        # Setup chemical systems...
        MolASolventSystem = OpenFEUtil.InitializeChemicalSystem(
            SmallMol=MolA, MacroMol=None, Solvent=Solvent, Name="%s_Solvent" % MolA.name
        )
        MolAVacuumSystem = OpenFEUtil.InitializeChemicalSystem(
            SmallMol=MolA, MacroMol=None, Solvent=None, Name="%s_Vacuum" % MolA.name
        )

        MolBSolventSystem = OpenFEUtil.InitializeChemicalSystem(
            SmallMol=MolB, MacroMol=None, Solvent=Solvent, Name="%s_Solvent" % MolB.name
        )
        MolBVacuumSystem = OpenFEUtil.InitializeChemicalSystem(
            SmallMol=MolB, MacroMol=None, Solvent=None, Name="%s_Vacuum" % MolB.name
        )

        # Setup MolASolvent to MolBSolvent transformation...
        TransformationProtocol = SetupTransformationProtocol(
            MolAToMolBMapping, RHFEProtocol, RHFEProtocolChargeCorrection
        )
        TransformationName = "%s_To_%s_Solvent" % (MolA.name, MolB.name)
        MolAToMolBSolventTransformation = OpenFEUtil.InitializeTransformation(
            StateA=MolASolventSystem,
            StateB=MolBSolventSystem,
            Mapping=MolAToMolBMapping,
            Protocol=TransformationProtocol,
            Name=TransformationName,
            Validate=False,
        )
        MolAToMolBTransformations.append(MolAToMolBSolventTransformation)

        # Setup MolAVacuum to MolBVacuum transformation...
        TransformationProtocol = RHFEProtocolVacuum
        TransformationName = "%s_To_%s_Vacuum" % (MolA.name, MolB.name)
        MolAToMolBVacuumTransformation = OpenFEUtil.InitializeTransformation(
            StateA=MolAVacuumSystem,
            StateB=MolBVacuumSystem,
            Mapping=MolAToMolBMapping,
            Protocol=TransformationProtocol,
            Name=TransformationName,
            Validate=False,
        )
        MolAToMolBTransformations.append(MolAToMolBVacuumTransformation)

    # Write out transformatios...
    WriteTransformations(MolAToMolBTransformations)

    return MolAToMolBTransformations


def SetupTransformationProtocol(MolAToMolBMapping, RHFEProtocol, RHFEProtocolChargeCorrection):
    """Setup transformation protocol."""

    from openfe.utils import ligand_utils

    ChargeDifference = ligand_utils.get_alchemical_charge_difference(MolAToMolBMapping)

    if ChargeDifference != 0:
        MolA = MolAToMolBMapping.componentA
        MolB = MolAToMolBMapping.componentB
        if OptionsInfo["RHFEChargeCorrection"]:
            TransformationProtocol = RHFEProtocolChargeCorrection
            MiscUtil.PrintInfo("")
            MiscUtil.PrintWarning(
                'The transformation between molecules %s and %s involves a charge change of %s. The RHFE setting parameters have been automatically updated for "Yes" value of option "--rhfeChargeCorrection" to employ a more expensive set of parameters specified by option "--rhfeChargeCorrectionParams". '
                % (MolA.name, MolB.name, ChargeDifference)
            )
        else:
            TransformationProtocol = RHFEProtocol
            MiscUtil.PrintInfo("")
            MiscUtil.PrintWarning(
                'The transformation between molecules %s and %s involves a charge change of %s. The RHFE setting parameters have not been automatically updated for "No" value of option "--rhfeChargeCorrection" to employ more expensive set of parameters specified by option "--rhfeChargeCorrectionParams". A word to the wise: You may want to consider sepecifying "Yes" value for option "--rhfeChargeCorrection".'
                % (MolA.name, MolB.name, ChargeDifference)
            )
    else:
        TransformationProtocol = RHFEProtocol

    return TransformationProtocol


def WriteTransformations(MolAToMolBTransformations):
    """Write out transformations."""

    TransformationsOutDirPath = pathlib.Path(OptionsInfo["TransformationsOutDirPath"])

    MiscUtil.PrintInfo(
        "Writing transformations files (Files: *.json; Count: %s; Subdirectory: %s)..."
        % (len(MolAToMolBTransformations), OptionsInfo["TransformationsOutDir"])
    )

    for Transformation in MolAToMolBTransformations:
        TransformationFilePath = TransformationsOutDirPath.joinpath("%s.json" % Transformation.name)
        Transformation.dump(TransformationFilePath)


def SetupProtocolDAGs(MolAToMolBTransformations):
    """Setup protocol Directed Acyclic Graphs (DAGs) for each transformation to
    to perform calculations.
    """

    MiscUtil.PrintInfo("\nSetting up protocol DAGs (Count: %s)..." % len(MolAToMolBTransformations))

    MolAToMolBProtocolDAGs = []
    for Transformation in MolAToMolBTransformations:
        ProtocolDAG = OpenFEUtil.InitializeProtocolDAG(Transformation, Name=Transformation.name)
        MolAToMolBProtocolDAGs.append(ProtocolDAG)

    return MolAToMolBProtocolDAGs


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


def ProcessProtocolResults(MolAToMolBTransformations, MolAToMolBProtocolResults):
    """Process protocol results."""

    ResultFileParams = OptionsInfo["ResultFileParams"]

    ResultFile = "%s_RHFE_Results.%s" % (OptionsInfo["OutfilePrefix"], ResultFileParams["Ext"])
    ResultFilePath = os.path.join(OptionsInfo["OutfileDirPath"], ResultFile)
    MiscUtil.PrintInfo("\nWriting %s..." % ResultFile)

    Precision = ResultFileParams["Precision"]

    ResultData = []
    for Index in range(0, len(MolAToMolBProtocolResults), 2):
        MolAToMolBSolventProtocolResult = MolAToMolBProtocolResults[Index]
        MolAToMolBVacuumProtocolResult = MolAToMolBProtocolResults[Index + 1]

        # Setup mol names using solvent transformation. The vacuum transformation
        # also contains the same pair of moleules.
        MolAToMolBSolventTransformation = MolAToMolBTransformations[Index]

        MolA = MolAToMolBSolventTransformation.stateA.components["ligand"]
        MolB = MolAToMolBSolventTransformation.stateB.components["ligand"]
        MolAName = MolA.name
        MolBName = MolB.name

        if MolAToMolBSolventProtocolResult is None or MolAToMolBVacuumProtocolResult is None:
            DeltaDeltaGHydration = "NA"
            DeltaDeltaGHydrationUncertainty = "NA"
        else:
            # Setup hyfration value without the units...
            MolAToMolBSolventDeltaG = MolAToMolBSolventProtocolResult.get_estimate()
            MolAToMolBVacuumDeltaG = MolAToMolBVacuumProtocolResult.get_estimate()

            DeltaDeltaGHydration = MolAToMolBSolventDeltaG.m - MolAToMolBVacuumDeltaG.m
            DeltaDeltaGHydration = "%.*f" % (Precision, DeltaDeltaGHydration)

            # Setup uncertainty value without the units...
            MolAToMolBSolventDeltaGUncertainty = MolAToMolBSolventProtocolResult.get_uncertainty()
            MolAToMolBVacuumDeltaGUncertainty = MolAToMolBVacuumProtocolResult.get_uncertainty()

            DeltaDeltaGHydrationUncertainty = np.sqrt(
                np.sum(np.square([MolAToMolBSolventDeltaGUncertainty.m, MolAToMolBVacuumDeltaGUncertainty.m]))
            )
            DeltaDeltaGHydrationUncertainty = "%.*f" % (Precision, DeltaDeltaGHydrationUncertainty)

        ResultData.append([MolAName, MolBName, DeltaDeltaGHydration, DeltaDeltaGHydrationUncertainty])

    ResultDF = pd.DataFrame(
        ResultData,
        columns=["MolAName", "MolBName", "DeltaDeltaG (MolA->MolB; RHFE) (kcal/mol)", "Uncertainty (kcal/mol)"],
    )
    ResultDF.to_csv(ResultFilePath, sep=ResultFileParams["Delim"], lineterminator="\n", index=False)


def GenerateAtomMappings(Mols):
    """Generate atom mappings."""

    MiscUtil.PrintInfo("\nChanging directory to %s..." % OptionsInfo["OutfileDir"])
    os.chdir(OptionsInfo["OutfileDirPath"])

    # Initialize atom mappers...
    MiscUtil.PrintInfo("\nInitializing atom mappers (%s)..." % " ".join(OptionsInfo["MapperList"]))
    Mappers = OpenFEUtil.InitializeAtomMappers(OptionsInfo["MapperList"], OptionsInfo["MapperParams"])

    MiscUtil.PrintInfo("\nInitializing atom mapper scorer (%s)..." % OptionsInfo["MapperScorer"])
    MapperScorer = OpenFEUtil.InitializeAtomMapperScorer(OptionsInfo["MapperScorer"])

    MolAToMolBMappings = None
    if OptionsInfo["MoleculePairsMode"]:
        MolAToMolBMappings = GenerateAtomMappingsForMoleculePairs(Mols, Mappers, MapperScorer)
    elif OptionsInfo["MoleculeNetworkMode"]:
        MolAToMolBMappings = GenerateAtomMappingForMoleculeNetwork(Mols, Mappers, MapperScorer)

    if MolAToMolBMappings is None or len(MolAToMolBMappings) == 0:
        MiscUtil.PrintError("Failed to generate atom mappings for small molecules.")

    return MolAToMolBMappings


def GenerateAtomMappingsForMoleculePairs(Mols, Mappers, MapperScorer):
    """Generate atom mappings for molecule pairs."""

    MiscUtil.PrintInfo(
        "\nGenerating atom mappings (%s: %d)..." % (OptionsInfo["Mode"], (len(OptionsInfo["MoleculePairsMolList"]) / 2))
    )

    # Generate atom mappings...
    MolAToMolBMappings = OpenFEUtil.SuggestAtomMappingsForMoleculePairs(
        OptionsInfo["MoleculePairsMolList"], Mappers, MapperScorer
    )

    # Write out image files...
    WriteMoleculePairsOutputFiles(MolAToMolBMappings)

    return MolAToMolBMappings


def WriteMoleculePairsOutputFiles(MolAToMolBMappings):
    """Write mapping image output files for molecule pairs."""

    if len(MolAToMolBMappings):
        MiscUtil.PrintInfo(
            "Writing molecule pairs output files (Files: <MolName1>_To_<MolName2>_*.png; Count: %s;  Subdirectory: %s)..."
            % (len(MolAToMolBMappings), OptionsInfo["PairImagesOutfileDir"])
        )

    for Mapping in MolAToMolBMappings:
        PairOutfilePath = SetupMappingImageFilePath("Molecule_Pair", Mapping, OptionsInfo["PairImagesOutfileDirPath"])
        OpenFEUtil.WriteMappingImageFile(Mapping, PairOutfilePath)


def GenerateAtomMappingForMoleculeNetwork(Mols, Mappers, MapperScorer):
    """Setup atom mapping for molecule network."""

    MiscUtil.PrintInfo("\nGenerating atom mappings (%s)..." % OptionsInfo["Mode"])

    # Generate network...
    LigandNetwork = OpenFEUtil.GenerateLigandNetwork(
        Mols, OptionsInfo["Network"], OptionsInfo["NetworkParams"], Mappers, MapperScorer
    )

    # Write out network output files...
    WriteMoleculeNetworkOutputFiles(LigandNetwork)

    # Setup mappings...
    MolAToMolBMappings = [Edge for Edge in LigandNetwork.edges]

    return MolAToMolBMappings


def WriteMoleculeNetworkOutputFiles(LigandNetwork):
    """Write out network output files."""

    NetworkName = OptionsInfo["Network"]
    MiscUtil.PrintInfo("\nGenerating ligand network (%s)..." % NetworkName)

    # Write out ligand network graphml and image files...
    NetworkOutfilePrefix = "%s_Network_%s_Mapper_%s" % (OptionsInfo["OutfilePrefix"], NetworkName, SetupMapperLabel())
    GraphMLOutfile = "%s.graphml" % NetworkOutfilePrefix
    ImageOutfile = "%s.%s" % (NetworkOutfilePrefix, OptionsInfo["NetworkParams"]["OutputNetworkFormat"])

    GraphMLOutfilePath = os.path.join(OptionsInfo["OutfileDirPath"], GraphMLOutfile)
    ImageOutfilePath = os.path.join(OptionsInfo["OutfileDirPath"], ImageOutfile)

    MiscUtil.PrintInfo("Writing %s..." % GraphMLOutfile)
    OpenFEUtil.WriteLigandNetworkGraphMLFile(LigandNetwork, GraphMLOutfilePath)

    MiscUtil.PrintInfo("Writing %s..." % ImageOutfile)
    OpenFEUtil.WriteLigandNetworkImageFile(LigandNetwork, ImageOutfilePath)

    #  Write out image files for edges...
    NetworkEdges = [Edge for Edge in LigandNetwork.edges]
    if OptionsInfo["NetworkParams"]["OutputEdges"]:
        if len(NetworkEdges):
            MiscUtil.PrintInfo(
                "Writing edge output files (Files: <MolName1>_To_<MolName2>_*.png; Count: %s; Subdirectory: %s)..."
                % (len(NetworkEdges), OptionsInfo["EdgeImagesOutfileDir"])
            )

        for Edge in NetworkEdges:
            EdgeOutfilePath = SetupMappingImageFilePath("Network", Edge, OptionsInfo["EdgeImagesOutfileDirPath"])
            OpenFEUtil.WriteMappingImageFile(Edge, EdgeOutfilePath)


def SetupMappingImageFilePath(ModeLabel, Mapping, OutfileDirPath):
    """Setup mapping image file path."""

    Outfile = "%s_To_%s_%s_Mapper_%s.png" % (
        Mapping.componentA.name,
        Mapping.componentB.name,
        ModeLabel,
        SetupMapperLabel(),
    )
    Outfile = re.sub(" ", "_", Outfile)

    OutfilePath = os.path.join(OutfileDirPath, Outfile)

    return OutfilePath


def SetupMapperLabel():
    """Setup mapper label."""

    return "_".join(OptionsInfo["MapperList"])


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

    if ValidMolCount < 2:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("Small molecule Input file must contain at least 2 molecules.\n")

    return Mols


def ValidateMoleculeNames(Mols):
    """Validate molecule names."""

    if OptionsInfo["MoleculePairsMode"]:
        OptionsInfo["MoleculePairsMolList"] = OpenFEUtil.ProcessMoleculePairs(Mols, OptionsInfo["MoleculePairsList"])
    elif OptionsInfo["MoleculeNetworkMode"]:
        if OptionsInfo["RadialNetworkStatus"]:
            OptionsInfo["NetworkParams"]["RadialCentralLigandMol"] = OpenFEUtil.ProcessRadialCentralLigandName(
                Mols, OptionsInfo["NetworkParams"]["RadialCentralLigand"]
            )


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
            'The small molecule input file contains molecules with missing partial charges. The execution of the script has been terminated for "Stop" value of "--missingChargedMode" option. You may continue the execution of the script by specifying "Calculate" value for "--missingChargedMode" option.\n\nThe missing charges will be automatically calculated by OpenFE RelativeHybridTopologyProtocol module during the calculation of RHFE. You may control the calculation of partial charges by specifying values for partialCharge* parameters using "--rhfeParams" option.  Alternatively, you may employ the OpenFECalculatePartialCharges.py script to calculate partial charges and use the small molecule input file containing charges to calculate RHFE.\n'
        )
    else:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintWarning(
            'The small molecule input file contains molecules with missing partial charges. The missing charges will be automatically calculated by OpenFE RelativeHybridTopologyProtocol module during the calculation of RHFE. You may control the calculation of partial charges by specifying values for partialCharge* parameters using "--rhfeParams" option. Alternatively, you may employ the OpenFECalculatePartialCharges.py script to calculate partial charges and use the small molecule input file containing charges to calculate RHFE.\n'
        )


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

    # Setup a images subdirectory for a network...
    EdgeImagesOutfileDir = "NetworkEdgeImages"
    EdgeImagesOutfileDirPath = os.path.join(OptionsInfo["OutfileDirPath"], EdgeImagesOutfileDir)
    if OptionsInfo["MoleculeNetworkMode"] and OptionsInfo["NetworkParams"]["OutputEdges"]:
        if not os.path.exists(EdgeImagesOutfileDirPath):
            os.mkdir(EdgeImagesOutfileDirPath)
    OptionsInfo["EdgeImagesOutfileDir"] = EdgeImagesOutfileDir
    OptionsInfo["EdgeImagesOutfileDirPath"] = EdgeImagesOutfileDirPath

    # Setup a images subdirectory for molecule pairs...
    PairImagesOutfileDir = "MoleculePairImages"
    PairImagesOutfileDirPath = os.path.join(OptionsInfo["OutfileDirPath"], PairImagesOutfileDir)
    if OptionsInfo["MoleculePairsMode"]:
        if not os.path.exists(PairImagesOutfileDirPath):
            os.mkdir(PairImagesOutfileDirPath)
    OptionsInfo["PairImagesOutfileDir"] = PairImagesOutfileDir
    OptionsInfo["PairImagesOutfileDirPath"] = PairImagesOutfileDirPath

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

    RHFESettings = openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol.default_settings()

    MiscUtil.PrintInfo("\nListing RHFE settings...")
    OpenFEUtil.ListOpenFESettings(RHFESettings)


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

    OptionsInfo["ExecuteDAGParams"] = OpenFEUtil.ProcessOptionOpenFEExecuteDAGParameters(
        "--executeDAGParams", Options["--executeDAGParams"]
    )

    OptionsInfo["LoggingLevel"] = Options["--loggingLevel"]

    OptionsInfo["MapperList"] = OpenFEUtil.ProcessOptionOpenFEMapper("-m, --mapper", Options["--mapper"])
    OptionsInfo["MapperParams"] = OpenFEUtil.ProcessOptionOpenFEMapperParameters(
        "-m, --mapperParams", Options["--mapperParams"]
    )
    OptionsInfo["MapperScorer"] = Options["--mapperScorer"]

    OptionsInfo["Mode"] = OpenFEUtil.ProcessOptionOpenFERelativeFreeEnergyMode("-m, --mode", Options["--mode"])
    OptionsInfo["MoleculePairsMode"] = True if re.match("^MoleculePairs$", OptionsInfo["Mode"], re.I) else False
    OptionsInfo["MoleculeNetworkMode"] = True if re.match("^MoleculeNetwork$", OptionsInfo["Mode"], re.I) else False

    OptionsInfo["MissingChargeMode"] = OpenFEUtil.ProcessOptionOpenFEMissingChargeMode(
        "--missingChargeMode", Options["--missingChargeMode"]
    )

    OptionsInfo["Network"] = OpenFEUtil.ProcessOptionOpenFENetwork("-n, --network", Options["--network"])
    OptionsInfo["RadialNetworkStatus"] = True if re.match("^Radial$", OptionsInfo["Network"], re.I) else False

    ParamsDefaultInfoOverride = {"OutputEdges": True}
    OptionsInfo["NetworkParams"] = OpenFEUtil.ProcessOptionOpenFENetworkParameters(
        "--networkParams",
        Options["--networkParams"],
        RadialNetworkStatus=OptionsInfo["RadialNetworkStatus"],
        ParamsDefaultInfo=ParamsDefaultInfoOverride,
    )

    OptionsInfo["MoleculePairs"] = Options["--moleculePairs"]
    OptionsInfo["MoleculePairsList"] = OpenFEUtil.ProcessOptionOpenFEMoleculePairs(
        "--moleculePairs", Options["--moleculePairs"]
    )
    OptionsInfo["MoleculePairsMolList"] = None

    OptionsInfo["ResultFileParams"] = OpenFEUtil.ProcessOptionOpenFEResultFileParameters(
        "--resultFileParams", Options["--resultFileParams"]
    )

    ParamsDefaultInfoOverride = {"EngineComputePlatform": "CPU"}
    OptionsInfo["RHFEParams"] = OpenFEUtil.ProcessOptionOpenFERelativeFreeEnergyParameters(
        "--rhfeParams", Options["--rhfeParams"], ParamsDefaultInfo=ParamsDefaultInfoOverride
    )

    OptionsInfo["RHFEChargeCorrection"] = True if re.match("^yes$", Options["--rhfeChargeCorrection"]) else False
    OptionsInfo["RHFEChargeCorrectionParams"] = (
        OpenFEUtil.ProcessOptionOpenFERelativeFreeEnergyChargeCorrectionParameters(
            "--rhfeChargeCorrectionParams", Options["--rhfeChargeCorrectionParams"]
        )
    )

    OptionsInfo["RHFEVacuumParams"] = OpenFEUtil.ProcessOptionOpenFERelativeFreeEnergyVacuumParameters(
        "--rhfeVacuumParams", Options["--rhfeVacuumParams"]
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

    for Mapper in Options["--mapper"].split(","):
        Mapper = Mapper.strip()
        MiscUtil.ValidateOptionTextValue("--mapper", Mapper, "LOMAP Kartograf")

    MiscUtil.ValidateOptionTextValue("--mapperScorer", Options["--mapperScorer"], "LOMAP")

    MiscUtil.ValidateOptionTextValue("-m, --mode", Options["--mode"], "MoleculePairs MoleculeNetwork")
    MiscUtil.ValidateOptionTextValue("--missingChargeMode", Options["--missingChargeMode"], "Calculate Stop")

    MoleculePairs = Options["--moleculePairs"]
    if not re.match("^auto$", MoleculePairs, re.I):
        MoleculePairsList = MoleculePairs.split(",")
        if len(MoleculePairsList) % 2:
            MiscUtil.PrintError(
                'The number of comma delimited values, %d, specified using "--moleculePairs" option must be an even number.'
                % (len(MoleculePairsList))
            )

    MiscUtil.ValidateOptionTextValue("-n, --network", Options["--network"], "LOMAP MinimalSpanning Radial")

    MiscUtil.ValidateOptionTextValue("--rhfeChargeCorrection", Options["--rhfeChargeCorrection"], "yes no")


# Setup a usage string for docopt...
_docoptUsage_ = """
OpenFECalculateRelativeHydrationFreeEnergy.py - Calculate relative hydration free energy

Usage:
    OpenFECalculateRelativeHydrationFreeEnergy.py [--executeDAGParams <Name,Value,..>] [--infileParams <Name,Value,...>]
                                                  [--loggingLevel <Info, Warning or Error>] [--mapper <mapper1, mapper2,...>] [--mapperParams <Name,Value,..>]
                                                  [--mapperScorer <LOMAP>] [--mode <MoleculePairs or MoleculeNetwork>] [--missingChargeMode <Calculate or Stop>]
                                                  [--moleculePairs <MolName1,MolName2,..>] [--network <text>] [--networkParams <Name,Value,..>]
                                                  [--outfilePrefix <text>] [--overwrite] [--rhfeParams <Name,Value,...>] [--rhfeChargeCorrection <yes or no>]
                                                  [--rhfeChargeCorrectionParams <Name,Value,...>] [--rhfeVacuumParams <Name,Value,...>]
                                                  [--resultFileParams <Name,Value,..>] [--solventParams <Name,Value,...>]
                                                  [-w <dir>] -i <infile>  -o <outifiledir>
    OpenFECalculateRelativeHydrationFreeEnergy.py -l | --list
    OpenFECalculateRelativeHydrationFreeEnergy.py -h | --help | -e | --examples

Description:
    Calculate Relative Hydration Free Energy (RHFE) for a pair of molecules in a
    small molecule input file. You may calculate RHFEs for specific pairs of
    molecules or all molecule pairs corresponding to edges in a molecule network.

    The small molecule input file must contain molecules already prepared for
    simulation. It must contain appropriate 3D coordinates along with no missing
    hydrogens.

    The MD simulation workflow, employed for the calculation of RHFEs, involves the
    following steps: initial minimization; NVT equilibration; NPT equilibration;
    production NPT. The MD simulation protocol is repeated 3 times for each pair
    pair of transformations, MolAToMolBSolvent and MolAToMolBVacuum,
    and the results are analyzed to estimate RHFEs. The default time and step size
    settings for the MD protocol are shown below:
        
        Protocol repeats, 3
        
        Time step size: 4.0 femtosecond
        
        Max minimization steps: 5,000
        
        NVT equilibration length: 1.0 nanosecond
        NPT equilibration length: 1.0 nanosecond
        NPT production length: 5.0 nanosecond
        
    Each MD simulation, by default, may run for 7 nanosecond, for a total of 21
    nanosecond to repeat it 3 times. The total MD simulation time for each pair
    of transformations, MolAToMolBSolvent and MolAToMolBVacuum,
    may correspond to more than 42 nanoseconds.

    The supported small molecule input file format are : SD (.sdf, .sd)

    Possible outfile prefix:
        
        <OutfilePrefix> or <InfileRoot>
        
    Possible output directories:
        
        <OutfileDir>
        
        <OutfileDir>/MoleculePairImages [ MoleculeNetwork mode ]
        <OutfileDir>/NetworkEdgeImages [ MoleculePairs mode]
        
        <OutfileDir>/Transformations
        <OutfileDir>/Results
        
    Possible output files and directories under <OutfileDir>:
        
        <OutfilePrefix>_RHFE_Results.<csv or tsv>
        
        MoleculePairImages/<MolAName>_To_<MolBName>_Molecule_Pair*.png
        ... ... ...
         
        <OutfilePrefix>_Network*.graphml
        <OutfilePrefix>_Network*.svg
        NetworkEdgeImages/<MolAName>_To_<MolBName>_Network*.png
        ... ... ...
        
        Transformations/<MolAName>_To_<MolBName>_Solvent.json
        Transformations/<MolAName>_To_<MolBName>_Vacuum.json
        ... ... ...
        
        Results/shared_RelativeHybridTopologyProtocolUnit-*/
        Results/scratch_RelativeHybridTopologyProtocolUnit-*/
        ... ... ...

Options:
    -e, --examples
        Print examples.
    --executeDAGParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for executing
        protocol DAGs (Directed Acyclic Graph) to run RHFE calculations.
        
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
        List default RHFE protocol settings provided by OpenFE module
        RelativeHybridTopologyProtocol.
    --loggingLevel <Info, Warning or Error>  [default: Error]
        Logging level to configure the 'root logger' via logging.basicConfig()
        function. The default logging level is changed from 'logging.INFO' to
        'logging.ERROR'. Otherwise, OpenFE and its associated modules
        may generate a lot of informational messages.
    --mapper <mapper1, mapper2>  [default: LOMAP]
        A comma delimited names of atom mappers for generating atom mapping
        corresponding to molecule pairs or edges in a molecule network. Possible
        values: LOMAP [ Lead Optimization MAPer; Ref 176 ] or Kartograf [ Ref 177 ].
        You may specify multiple mappers for generating mapping between pair of
        molecules. All specified mappers are employed to identify the highest
        scoring mapping between a pair of molecules.
    --mapperParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for atom mappers
        employed to generate mapping between molecule pairs or edges in a molecule
        network. 
        
        The supported parameter names along with their default values are
        are shown below:
            
            lomapTime, 20, [ Units: seconds ]
            lomapThreeD, yes [ Possible values: yes or no ]
            lomapMax3D, 1.0 [ Units: Angstrom ]
            lomapElementChange, yes [ Possible values: yes or no]
            lomapSeed, None [ Possible value: A string. An empty string causes
                MCS search to start from scratch ]
            lomapShift, no [  Possible values: yes or no]
            
            kartografAtomMaxDistance, 0.95 [ Units: Angstrom ]
            kartografAtomMapHydrogens, yes [ Possible values: yes or no ]
            kartografMapHydrogensOnHydrogensOnly, No [ Possible values: yes or
                no ]
            kartografMapExactRingMatchesOnly, yes [ Possible values: yes or no ]
            kartografAllowPartialFusedRings, yes [ Possible values: yes or no ]
            
        A brief description of parameters is provided below:
            
            lomapTime: Time out for MCS algorithm.
            lomapThreeD: Use atom positions to prune symmetric mappings.
            lomapMax3D: Forbid mapping between atoms with distance more than
                specified value.
            lomapElementChange: Allow mappings that change an atom element.
            lomapSeed: An Empty SMARTS string causes MCS search to start from
                scratch.
            lomapShift: Keep pre-aligned atom positions for 3D position checks.
            
            kartografAtomMaxDistance: Geometric criteria for two atoms
                corresponding to maximum distance between them.
            kartografAtomMapHydrogens: Map hydrogens.
            kartografMapHydrogensOnHydrogensOnly: Map hydrogens only on
                hydrogens.
            kartografMapExactRingMatchesOnly: Map rings with only matching ring
                size and bond orders. In addition, ring breaking is not
                permitted.
            kartografAllowPartialFusedRings: Allow mapping of partially fused
                rings.
            
    --mapperScorer <LOMAP>  [default: LOMAP]
        Atom mapper scorer to use for scoring mapping between molecule pairs or
        edges in a molecule network. Possible value: LOMAP. The atom scorer is
        not used during the generation of MinimalSpanning network.
    -m, --mode <MoleculePairs or MoleculeNetwork>  [default: MoleculePairs]
        Calculate RHFEs for specified pairs of molecules or all molecule pairs
        corresponding to edges in a molecule network.
    --missingChargeMode <Calculate or Stop>  [default: Stop]
        Calculate missing partial charges for molecules before running RHFE
        calculations or terminate the execution of the script. The missing
        partial charges will be automatically calculated by OpenFE module
        RelativeHybridTopologyProtocol during the calculation of RHFE. You
        may control the calculation of partial charges by specifying values for
        partialCharge* parameters using '--rhfeParams' option.
    --moleculePairs <MolName1,MolName2,..>  [default: auto]
        A comma delimited list of molecule name pairs for calculating RHFEs.
        Default: the names of the first and second molecule in small molecule
        input file. This option is only used during 'MoleculePairs' value for
        '-m, --mode' option. 
    -n, --network <text>  [default: MinimalSpanning]
        Name of a molecule network to generate for calculating RHFEs. Possible
        values: LOMAP, MinimalSpanning or Radial. This option is only used during
        'MoleculeNetwork' value for '-m, --mode' option. 
    --networkParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        a molecule network.
        
        The supported parameter names along with their default values are
        are shown below:
            
            lomapDistanceCutoff, 0.4
            lomapMaxPathLength, 6
            lomapRequireCycleCovering, yes  [ Possible values: yes or no ]
            
            minimalSpanningProgress, no  [ Possible values: yes or no ]
            
            radialCentralLigand, None  [ Possible values: Valid ligand name ]
            
            outputEdges, no  [ Possible values: yes or no ]
            outputNetworkFormat, svg  [ Possible values: Any valid format. ]
            
        A brief description of parameters is provided below:
            
            lomapDistanceCutoff: Maximum distance/dissimilarity between two
                molecules for an edge to be accepted.
            lomapMaxPathLength: Maximum distance between any two molecules in
                the resulting network
            lomapRequireCycleCovering: Add cycles into the network
            
            minimalSpanningProgress: Show progress using tqdm.
            
            radialCentralLigand: Name of central ligand. A valid ligand name
                must be specified to generate a radial molecule network.
            
            outputEdges: Generate PNG image files for all edges in a molecule
                network.
            outputNetworkFormat: Valid image file format for molecule network.
                You must specify a valid format supported by Python module
                Matplotlib. For example: PNG (.png), SVG (.svg), PDF (.pdf),
                etc. In addition, the graphml file is always generated.
            
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
            
    -r, --rhfeParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for RHFE protocol
        settings employed during the calculation of RHFEs.
        
        The default values are automatically updated to match settings provided by
        OpenFE module RelativeHybridTopologyProtocol.
        
        You must specify valid OpenFE values for these parameters. An extensive
        validation is not performed.
        
        The supported parameter names along with their default values are
        are shown below:
            
            protocolRepeats, 3
            
            Alchemical settings:
            
            alchemicalEndstateDispersionCorrection, no  [ Possible values:
                yes or no ]
            alchemicalExplicitChargeCorrection, no  [ Possible values:
                yes or no ]
            alchemicalExplicitChargeCorrectionCutoff, 0.8  [ Units: nanometer ]
            alchemicalSoftcoreLJ, Gapsys [ Possible values: Gapsys or Beutler ] 
            alchemicalSoftcoreAlpha, 0.85
            alchemicalTurnOffCoreUniqueExceptions, no  [ Possible values:
                yes or no ]
            alchemicalUseDispersionCorrection, no [ Possible values: yes or no ]
            
            Engine settings:
            
            engineComputePlatform, CPU  [ Possible values: CPU, CUDA, OpenCL,
                or Reference ]
            engineGpuDeviceIndex, None [ Possible values: 0, 0 1, etc. ]
            
            Forcefield settings:
            
            forcefieldConstraints, HBonds  [ Possible values: HBonds, ALLBonds or
                HAngles  ]
            forcefields, amber/ff14SB.xml amber/tip3p_standard.xml
                amber/tip3p_HFE_multivalent.xml amber/phosaa10.xml
                [ Possible values: A space delimited list of valid names. ]
            forcefieldHydrogenMass, 3.0  [ Units: amu ]
            forcefieldNonbondedCutoff, 0.9  [ Units: nanometer ]
            forcefieldNonbondedMethod, PME [ Possible values: PME or NoCutoff ]
            forcefieldRigidWater, yes  [ Possible values: yes or no ]
            forcefieldSmallMoleculeForcefield, openff-2.1.1  [ Possible value:
                A valid forcefield name. ]
            
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
            
            lambdaFunctions, default  [ Possible values: Default, namd, or
                quarters ]
            lambdaWindows, 11
            
            Output settings:
            
            outputCheckpointInterval, 1.0 [ Units: nanosecond ]
            outputCheckpointStorageFilename, checkpoint.chk
            outputForcefieldCache, db.json
            outputFilename, simulation.nc
            outputIndices, not water  [ Possible value: Any valid selection. ]
            outputStructure, hybrid_system.pdb
            outputPositionsWriteFrequency, 100.0 [ Units: picosecond ]
            outputVelocitiesWriteFrequency, None  [  Possible values: > 0;
                Units: picosecond ]
            
            Partial charge settings:
            
            partialChargeNaglModel, None  [ Default: Production AM1BCC model for
                NAGL; Possible value: Any valid name. ]
            partialChargeNumberOfConformers, None  [ Possible value: > 0 ]
            partialChargeOffToolkitBackend, AmberTools  [ Possible values:
                AmberTools or RDKit ]
            partialChargeMethod, AM1BCC  [ Possble values: AM1BCC, Espaloma,
                or NAGL ]
            
            Simulation settings:
            
            simulationEarlyTerminationTargetError, 0.0 [ Units:
                kilocalorie_per_mole ]
            simulationEquilibrationLength, 1.0 [ Units: nanosecond ]
            simulationMinimizationSteps, 5000
            simulationNReplicas, 11
            simulationProductionLength, 5.0 [ Units: nanosecond ]
            simulationRealTimeAnalysisInterval, 250.0 [ Units: picosecond ]
            simulationRealTimeAnalysisMinimumTime, 500.0  [ Units: picosecond ]
            simulationSamplerMethod, repex  [ Possible values: repex, sams,
                or independent ]
            simulationSamsFlatnessCriteria, logZ-flatness  [ Possible values:
                logZ-flatness, minimum-visits or histogram-flatness ]
            simulationSamsGamma0, 1.0
            simulationTimePerIteration, 2.5  [ Units: picosecond ]
            
            Solvation settings:
            
            solvationBoxShape, dodecahedron  [  Possible values: cube,
                dodecahedron, or octahedron ]
            solvationBoxSize, None  [ Possible value: A triplet of space
                X Y Z values; Units: nanometer ]
            solvationSolventModel, tip3p  [ Possible values: tip3p, spce, tip4pew,
                or tip5p ]
            solvationSolventPadding, 1.5  [ Units: nanometer ]
            
            Thermo settings:
            
            thermoPh, None  [ Possible values: > 0 ]
            thermoPressure, 1.0 [ Units: bar ]
            thermoRedoxPotential, None  [ Possible values: A valid float.
                Units: millivolts (mV) ]
            thermoTemperature, 298.15  [ Units: kelvin ]
            
        A brief description of parameters, taken from OpenFE documentation, is
        provided below:
            
            protocolRepeats: Number of completely independent repeats of the
                entire sampling process.
            
            Alchemical settings:
            
            Parameters controlling the creation of the hybrid topology system,
            including various parameters ranging from softcore parameters to
            whether or not to apply an explicit charge correction for systems
            with net charge changes.
            
            alchemicalEndstateDispersionCorrection: Employ extra unsampled
                endstate windows for long range correction.
            alchemicalExplicitChargeCorrection: Explicitly account for a charge
                difference during the alchemical transformation by transforming
                a water to a counterion of the opposite charge of the formal
                charge difference.
            alchemicalExplicitChargeCorrectionCutoff: Minimum distance from the
                system solutes from which an alchemical water can be chosen.
            alchemicalSoftcoreLJ: Use LJ softcore function as defined by Gapsys
                [ Ref 181 ] or Buetler [ Ref 182 ].
            alchemicalSoftcoreAlpha: Softcore alpha parameter.
            alchemicalTurnOffCoreUniqueExceptions: Turn off interactions for
                new exceptions (not just 1,4s) at lambda 0 and old exceptions at
                lambda 1 between unique atoms and core atoms.
            alchemicalUseDispersionCorrection: Use dispersion correction in the
                hybrid topology state.
        
            Engine settings:
            
            Parameters configuring the compute platform used by the OpenMM to
            perform the simulation.
            
            engineComputePlatform: Platform to use for running OpenMM MD
                calculations.
            engineGpuDeviceIndex: Space delimited list of device indices to use
                for running OpenMM MD calculations.
            
            Forcefield settings:
            
            Parameters to set up the force field with OpenMM Force Fields,
            including the general force fields, the small molecule force field,
            the nonbonded method, and the nonbonded cutoff.
            
            forcefieldConstraints: Constraints to use.
            forcefields: List of valid forcefield paths for all components
                except small molecules.
            forcefieldHydrogenMass: Mass to be repartitioned to hydrogens from
                neighboring heavy atoms.
            forcefieldNonbondedCutoff: Cutoff for short range nonbonded
                interactions.
            forcefieldNonbondedMethod: Method for treating nonbonded
                interactions.
            forcefieldRigidWater: Use a rigid water model.
            forcefieldSmallMoleculeForcefield: A valid forcefield name to use
                for small molecules.
            
            Integrator settings
            
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
            
            Lambda protocol parameters, including number of lambda windows and
            lambda functions.
            
            lambdaFunctions: Function name to use for alchemical mutation.
            lambdaWindows: Number of lambda windows to calculate.
            
            Output settings:
            
            Parameter controlling simulation output, including the frequency to
            write a checkpoint file, the selection string for writing selected
            coordinates, and the paths to the trajectory and output structure
            files.
            
            outputCheckpointInterval: Frequency to write the checkpoint file.
            outputCheckpointStorageFilename: Checkpoint filename.
            outputForcefieldCache: Filename for caching small molecule residue
                templates.
            outputFilename: Trajectory filename.
            outputIndices: Selection string for selecting coordinates to write.
            outputStructure: Hybrid topology structure filename.
            outputPositionsWriteFrequency: Frequency for writing positions to
                trajectory file.
            outputVelocitiesWriteFrequency: Frequency for writing velocities to
                trajectory file.
            
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
            
            Simulation settings:
            
            Parameters controlling the simulation plan and the alchemical
            sampler, including the number of minimization steps, lengths of
            equilibration and production runs, the sampler method (e.g.
            Hamiltonian REPlica EXchange (repex), and the time interval at
            which to perform an analysis of the free energies.
            
            simulationEarlyTerminationTargetError: Target error for the real
                time analysis measured in kcal/mol. Once the MBAR error of the
                free energy is at or below this value, the simulation will be
                considered complete. The suggested value of 0.12 has shown to
                be effective in both hydration and binding free energy
                benchmarks.
            simulationEquilibrationLength: Length of the equilibration phase.
                The specified value must be divisible by 'integratorTimestep'.
            simulationMinimizationSteps: Maximum number of minimization steps
                to perform.
            simulationNReplicas: Number of replicas to use.
            simulationProductionLength: Length of the production phase.
                The specified value must be divisible by 'integratorTimestep'.
            simulationRealTimeAnalysisInterval: Time interval for performing
                analysis of the free energies. At each interval, real time
                analysis data will be written to a yaml file named
                <outputFileName>_real_time_analysis.yaml. The current error
                in the estimate will also be assessed and the simulation will
                be terminated when it drops below
                'simulationEarlyTerminationTargetError'.
            simulationRealTimeAnalysisMinimumTime: Minimum simulation time
                after which the real time analysis is performed.
            simulationSamplerMethod: Alchemical sampling method to use:
                REPEX (Hamiltonian REPlica EXchange), SAMS (Self-Adjusted
                Mixture Sampling), or Independent (Independently sampled lambda
                windows).
            simulationSamsFlatnessCriteria:Method for assessing when to switch
                to asymptomatically optimal scheme for SAMS.
            simulationSamsGamma0: Initial weight adaptation rate for SAMS.
            simulationTimePerIteration: Simulation time between each MCMC move
               attempt 
            
            Solvation settings:
            
            Solvation parameters for the system, including the solvent model and
            the solvent padding.
            
            solvationBoxShape: Shape of the periodic solvent box to create.
            solvationBoxSize: Lengths of the unit cell for a solvent box.
            solvationSolventModel: Forcefield water model to use during
                solvation and defining the model properties.
            solvationSolventPadding: Minimum distance from any solute bounding
                sphere to the edge of the box.
            
            Thermo settings:
            
            Thermodynamic parameters, including the temperature and the pressure
            of the system.
            
            thermoPh: Simulation pH
            thermoPressure: Simulation pressure.
            thermoRedoxPotential:Simulation redox potential.
            thermoTemperature: Simulation temperature. 
            
    --rhfeChargeCorrection <yes or no>  [default: yes]
        Perform automatic charge correction for charge changing transformations
        during the calculation of RHFEs. The '--rhfeChargeCorrectionParams' are
        used during the automatic charge correction to override the corresponding
        values in '-r, --rhfeParams'.
    --rhfeChargeCorrectionParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs to use for RHFE
        protocol settings during explicit charge correction for charge changing
        transformation between pair of molecules. These parameters override the
        corresponding values in '-r, --rhfeParams'.
        
        The default parameter values for charge changing transformations are based
        on the industry benchmarking performed by OpenFE.
        
        The supported parameter names along with their default values are
        are shown below:
            
            alchemicalExplicitChargeCorrection, yes [ Possible values: yes or no ]
            simulationProductionLength, 20 [ Units: nanosecond ]
            simulationNReplicas, 22
            lambdaWindows, 22
            
        A brief description of these parameters is available under the corresponding
        parameters in the section for '-r, --rhfeParams'.
    --rhfeVacuumParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs to use for RHFE
        protocol settings during transformations in vacuum. These parameters
        override the corresponding values in '-r, --rhfeParams'.
        
        The supported parameter names along with their default values are
        are shown below:
            
            forcefieldNonbondedMethod, NoCutoff [ Possible values: PME or
                NoCutoff ]
            
        A brief description of these parameters is available under the corresponding
        parameters in the section for '-r, --rhfeParams'.
    --solventParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for solvent
        component. You must specify valid OpenFE values. No extensive validation
        is performed. These parameters are used in conjunction with solvation*
        parameters available through '--rhfeParams' to perform solvation.
        
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

    To calculate RHFE for a pair molecules corresponding to the fist and second
    molecules in a SD file, performing 3 independent repeats of the entire MD sampling
    process to  estimate RHFE for a pair of molecules, each MD repeat consisting of
    minimization (5,000 steps) followed by NVT and NPT equilibration (1 ns;
    250,000 steps) leading to NPT production (5s; 1,250,000 steps) using a step size
    of 4 fs, writing out appropriate trajectory and PDB files for each MD repeat
    in Results subdirectory under output directory, generating final results file
    along with appropriate graph and image files under output directory, type:

        % OpenFECalculateRelativeHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsRHFE

    To run the first example for calculating RHFE for a specific pair molecules
    using CUDA platform on your machine to perform MD simulations and generate
    various output files, type:

        % OpenFECalculateRelativeHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsRHFE -m MoleculePairs
          --moleculePairs "lig_ejm_31, lig_ejm_47"
          --rhfeParams "engineComputePlatform,CUDA"

    To run the second example to see all warning messages produced by OpenFE
    modules and write various output files, type;

        % OpenFECalculateRelativeHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsRHFE -m MoleculePairs
          --moleculePairs "lig_ejm_31, lig_ejm_47"
          --rhfeParams "engineComputePlatform,CUDA"
          --loggingLevel Warning

    To run the first example for calculating RHFE for all pairs of molecules
    corresponding to edges in a molecule network using CUDA platform on your
    machine to perform MD simulations and generate various output files, type:

        % OpenFECalculateRelativeHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsRHFE -m MoleculeNetwork
          --network MinimalSpanning
          --rhfeParams "engineComputePlatform,CUDA"

    To run the first example for calculating RHFE for a specific pair molecules
    using CUDA platform on your machine to perform MD simulations, automatically
    calculate missing partial charges for molecules, and generate various output
    files, type:

        % OpenFECalculateRelativeHydrationFreeEnergy.py
          -i SampleTyk2LigandsNoCharges.sdf -o SampleTyk2LigandsRHFE
          -m MoleculePairs --moleculePairs "lig_ejm_31, lig_ejm_47"
          --rhfeParams "engineComputePlatform,CUDA"
          --missingChargeMode Calculate

    To run the second example by specifying explict values for various parametres
    and generate various output files, type:

        % OpenFECalculateRelativeHydrationFreeEnergy.py
          -i SampleTyk2Ligands.sdf -o SampleTyk2LigandsRHFE -m MoleculePairs
          --moleculePairs "lig_ejm_31, lig_ejm_47"
          --loggingLevel Error
          --executeDAGParams "keepShared, yes, nRetries, 2" --mapper LOMAP
          --mapperParams "lomapTime, 20, lomapThreeD, yes"
           --missingChargeMode Stop --rhfeParams "protocolRepeats,3,
          alchemicalSoftcoreLJ, Gapsys, engineComputePlatform,CUDA,
          forcefieldConstraints, HBonds, forcefieldHydrogenMass, 3.0,
          forcefieldNonbondedMethod, PME, integratorTimestep, 4.0,
          lambdaWindows, 11, outputCheckpointInterval, 250.0,
          simulationMinimizationSteps, 5000, simulationEquilibrationLength, 1.0,
          simulationProductionLength, 5.0, solvationBoxShape, cube,
          solvationSolventPadding, 1.2, thermoPressure, 0.98692327,
          thermoTemperature, 298.15"
          --solventParams "positiveIon, Na+, negativeIon, Cl-"

Author:
    Manish Sud(msud@san.rr.com)

See also:
    OpenFECalculateAbsoluteBindingFreeEnergy.py,
    OpenFECalculateAbsoluteHydrationFreeEnergy.py, OpenFECalculatePartialCharges.py,
    OpenFECalculateRelativeBindingFreeEnergy.py, OpenFEGenerateLigandNetwork.py

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
