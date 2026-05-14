#!/usr/bin/env python
#
# File: VinaPerformDocking.py
# Author: Manish Sud <msud@san.rr.com>
#
# Acknowledgments: Diogo Santos-Martins and Stefano Forli
#
# Copyright (C) 2026 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using AutoDockVina
# and Meeko, open source software packages for docking, and RDKit, an open
# source toolkit for cheminformatics developed by Greg Landrum.
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
import tempfile
import json
import glob
import multiprocessing as mp

# AutoDock Vina imports...
try:
    from vina import Vina
    from vina import __version__ as vinaVersion
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import AutoDock Vina module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your Vina environment and try again.\n\n")
    sys.exit(1)

# AutoDock Meeko imports...
try:
    from meeko import __version__ as meekoVersion
    from meeko import MoleculePreparation
    from meeko import PDBQTMolecule
    from meeko import RDKitMolCreate
    from meeko import PDBQTWriterLegacy
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import AutoDock Meeko module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your Meeko environment and try again.\n\n")
    sys.exit(1)

# RDKit imports...
try:
    from rdkit import rdBase
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import RDKit module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your RDKit environment and try again.\n\n")
    sys.exit(1)

# MayaChemTools imports...
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), "..", "lib", "Python"))
try:
    from docopt import docopt
    import MiscUtil
    import RDKitUtil
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
        "\n%s (Vina v%s; Meeko v%s; RDKit v%s; MayaChemTools v%s; %s): Starting...\n"
        % (
            ScriptName,
            vinaVersion,
            meekoVersion,
            rdBase.rdkitVersion,
            MiscUtil.GetMayaChemToolsVersion(),
            time.asctime(),
        )
    )

    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()

    # Retrieve command line arguments and options...
    RetrieveOptions()

    # Process and validate command line arguments and options...
    ProcessOptions()

    # Perform actions required by the script...
    PerformDocking()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def PerformDocking():
    """Perform docking."""

    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])

    # Set up molecule writers...
    Writer, WriterFlexRes = SetupWriters()
    if WriterFlexRes is None:
        MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["Outfile"])
    else:
        MiscUtil.PrintInfo("Generating files %s and %s..." % (OptionsInfo["Outfile"], OptionsInfo["OutfileFlexRes"]))

    MolCount, ValidMolCount, DockingFailedCount = ProcessMolecules(Mols, Writer, WriterFlexRes)

    CloseWriters(Writer, WriterFlexRes)

    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during docking: %d" % DockingFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount + DockingFailedCount))


def ProcessMolecules(Mols, Writer, WriterFlexRes):
    """Process and dock molecules."""

    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols, Writer, WriterFlexRes)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols, Writer, WriterFlexRes)


def ProcessMoleculesUsingSingleProcess(Mols, Writer, WriterFlexRes):
    """Process and dock molecules using a single process."""

    VinaHandle = InitializeVina(OptionsInfo["QuietMode"])
    MolPrepHandle = InitializeMeekoMolPrepration(OptionsInfo["QuietMode"])
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("\nVina configuration:\n%s" % VinaHandle)

    MiscUtil.PrintInfo("\nDocking molecules (Mode: %s)..." % OptionsInfo["Mode"])

    (MolCount, ValidMolCount, DockingFailedCount) = [0] * 3
    for Mol in Mols:
        MolCount += 1

        if Mol is None:
            continue

        if not CheckAndValidateMolecule(Mol, MolCount):
            continue

        ValidMolCount += 1
        CalcStatus, PoseMol, PoseMolEnergies, PoseMolFlexRes = DockMolecule(VinaHandle, MolPrepHandle, Mol, MolCount)

        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning("Failed to dock  molecule %s" % RDKitUtil.GetMolName(Mol, MolCount))

            DockingFailedCount += 1
            continue

        WriteMolPoses(Writer, Mol, MolCount, PoseMol, PoseMolEnergies, WriterFlexRes, PoseMolFlexRes)

    return (MolCount, ValidMolCount, DockingFailedCount)


def ProcessMoleculesUsingMultipleProcesses(Mols, Writer, WriterFlexRes):
    """Process and calculate energy of molecules using multiprocessing."""

    MiscUtil.PrintInfo("\nDocking molecules using multiprocessing...")
    MiscUtil.PrintInfo("\nDocking molecules (Mode: %s)..." % OptionsInfo["Mode"])

    MPParams = OptionsInfo["MPParams"]

    # Setup data for initializing a worker process...
    InitializeWorkerProcessArgs = (
        MiscUtil.ObjectToBase64EncodedString(Options),
        MiscUtil.ObjectToBase64EncodedString(OptionsInfo),
    )

    # Setup a encoded mols data iterable for a worker process...
    WorkerProcessDataIterable = RDKitUtil.GenerateBase64EncodedMolStrings(Mols)

    # Setup process pool along with data initialization for each process...
    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo(
            "\nConfiguring multiprocessing using %s method..."
            % ("mp.Pool.imap()" if re.match("^Lazy$", MPParams["InputDataMode"], re.I) else "mp.Pool.map()")
        )
        MiscUtil.PrintInfo(
            "NumProcesses: %s; InputDataMode: %s; ChunkSize: %s\n"
            % (
                MPParams["NumProcesses"],
                MPParams["InputDataMode"],
                ("automatic" if MPParams["ChunkSize"] is None else MPParams["ChunkSize"]),
            )
        )

    ProcessPool = mp.Pool(MPParams["NumProcesses"], InitializeWorkerProcess, InitializeWorkerProcessArgs)

    # Start processing...
    if re.match("^Lazy$", MPParams["InputDataMode"], re.I):
        Results = ProcessPool.imap(WorkerProcess, WorkerProcessDataIterable, MPParams["ChunkSize"])
    elif re.match("^InMemory$", MPParams["InputDataMode"], re.I):
        Results = ProcessPool.map(WorkerProcess, WorkerProcessDataIterable, MPParams["ChunkSize"])
    else:
        MiscUtil.PrintError(
            'The value, %s, specified for "--inputDataMode" is not supported.' % (MPParams["InputDataMode"])
        )

    (MolCount, ValidMolCount, DockingFailedCount) = [0] * 3
    for Result in Results:
        MolCount += 1

        MolIndex, EncodedMol, CalcStatus, EncodedPoseMol, PoseMolEnergies, EncodedPoseMolFlexRes = Result

        if EncodedMol is None:
            continue

        ValidMolCount += 1

        if not CalcStatus:
            if not OptionsInfo["QuietMode"]:
                Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
                MolName = RDKitUtil.GetMolName(Mol, MolCount)
                MiscUtil.PrintWarning("Failed to dock molecule %s" % MolName)

            DockingFailedCount += 1
            continue

        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
        PoseMol = None if EncodedPoseMol is None else RDKitUtil.MolFromBase64EncodedMolString(EncodedPoseMol)
        PoseMolFlexRes = (
            None if EncodedPoseMolFlexRes is None else RDKitUtil.MolFromBase64EncodedMolString(EncodedPoseMolFlexRes)
        )

        WriteMolPoses(Writer, Mol, MolCount, PoseMol, PoseMolEnergies, WriterFlexRes, PoseMolFlexRes)

    return (MolCount, ValidMolCount, DockingFailedCount)


def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""

    global Options, OptionsInfo

    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())

    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Initialize in a quiet mode...
    OptionsInfo["VinaHandle"] = InitializeVina(True)
    OptionsInfo["MolPrepHandle"] = InitializeMeekoMolPrepration(True)


def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""

    MolIndex, EncodedMol = EncodedMolInfo

    CalcStatus, PoseMol, PoseMolEnergies, PoseMolFlexRes = [False, None, None, None]

    if EncodedMol is None:
        return [MolIndex, None, CalcStatus, PoseMol, PoseMolEnergies, PoseMolFlexRes]

    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    MolCount = MolIndex + 1

    if not CheckAndValidateMolecule(Mol, MolCount):
        return [MolIndex, None, CalcStatus, PoseMol, PoseMolEnergies, PoseMolFlexRes]

    CalcStatus, PoseMol, PoseMolEnergies, PoseMolFlexRes = DockMolecule(
        OptionsInfo["VinaHandle"], OptionsInfo["MolPrepHandle"], Mol, MolCount
    )

    EncodedPoseMol = (
        None
        if PoseMol is None
        else RDKitUtil.MolToBase64EncodedMolString(
            PoseMol, PropertyPickleFlags=Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps
        )
    )

    EncodedPoseMolFlexRes = (
        None
        if PoseMolFlexRes is None
        else RDKitUtil.MolToBase64EncodedMolString(
            PoseMolFlexRes,
            PropertyPickleFlags=Chem.PropertyPickleOptions.MolProps | Chem.PropertyPickleOptions.PrivateProps,
        )
    )

    return [MolIndex, EncodedMol, CalcStatus, EncodedPoseMol, PoseMolEnergies, EncodedPoseMolFlexRes]


def DockMolecule(VinaHandle, MolPrepHandle, Mol, MolNum=None):
    """Dock molecule."""

    Status, PoseMol, PoseMolEnergies, PoseMolFlexRes = [False, None, None, None]

    if OptionsInfo["VinaVerbosity"] != 0:
        MiscUtil.PrintInfo("\nProcessing molecule %s..." % RDKitUtil.GetMolName(Mol, MolNum))

    PDBQTMolStr = PrepareMolecule(MolPrepHandle, Mol)
    if PDBQTMolStr is None:
        return (Status, PoseMol, PoseMolEnergies, PoseMolFlexRes)
    VinaHandle.set_ligand_from_string(PDBQTMolStr)

    if OptionsInfo["ScoreOnlyMode"]:
        return ProcessMoleculeForScoreOnlyMode(VinaHandle, Mol)
    elif OptionsInfo["LocalOptimizationOnlyMode"]:
        return ProcessMoleculeForLocalOptimizationOnlyMode(VinaHandle, Mol)
    elif OptionsInfo["DockMode"]:
        return ProcessMoleculeForDockMode(VinaHandle, Mol)
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "-m, --mode" is not valid. Supported values: Dock LocalOptimizationOnly ScoreOnly'
            % OptionsInfo["Mode"]
        )

    return (Status, PoseMol, PoseMolEnergies, PoseMolFlexRes)


def ProcessMoleculeForScoreOnlyMode(VinaHandle, Mol):
    """Score molecule without performing any docking."""

    Status, PoseMol, Energies, PoseMolFlexRes = [False, None, None, None]

    # Score molecule...
    try:
        Energies = VinaHandle.score()
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to score molecule:\n%s\n" % ErrMsg)
        return (False, None, None, None)

    if len(Energies) == 0:
        return (False, None, None, None)

    Status = True
    PoseMol = Mol

    return (Status, PoseMol, Energies, PoseMolFlexRes)


def ProcessMoleculeForLocalOptimizationOnlyMode(VinaHandle, Mol):
    """Score molecule after a local optimization wthout any docking."""

    Status, PoseMol, Energies, PoseMolFlexRes = [False, None, None, None]

    # Optimize and score molecule...
    try:
        Energies = VinaHandle.optimize()
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to perform local optimization:\n%s\n" % ErrMsg)
        return (False, None, None, None)

    if len(Energies) == 0:
        return (False, None, None, None)

    # Write optimize pose to a temporray file and retrieve the PDBQT string for the pose...
    (_, TmpFile) = tempfile.mkstemp(suffix=".pdbqt", prefix="VinaOptimize_", text=True)
    VinaHandle.write_pose(TmpFile, overwrite=True)

    TmpFH = open(TmpFile, "r")
    PosePDBQTOutputStr = TmpFH.read()
    TmpFH.close()

    os.remove(TmpFile)

    # Setup a mol containing optimized pose...
    try:
        PosePDBQTOutputMol = PDBQTMolecule(PosePDBQTOutputStr)
        PoseMols = RDKitMolCreate.from_pdbqt_mol(PosePDBQTOutputMol)
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to retrieve optimize pose:\n%s\n" % ErrMsg)
        return (False, None, None, None)

    if len(PoseMols) == 0:
        return (False, None, None, None)

    Status = True
    PoseMol = PoseMols[0]

    return (Status, PoseMol, Energies, PoseMolFlexRes)


def ProcessMoleculeForDockMode(VinaHandle, Mol):
    """Dock and score molecule."""

    Status, PoseMol, Energies, PoseMolFlexRes = [False, None, None, None]

    # Dock molecule...
    try:
        VinaHandle.dock(
            exhaustiveness=OptionsInfo["Exhaustiveness"],
            n_poses=OptionsInfo["NumPoses"],
            min_rmsd=OptionsInfo["MinRMSD"],
            max_evals=OptionsInfo["MaxEvaluations"],
        )
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to dock molecule:\n%s\n" % ErrMsg)
        return (False, None, None, None)

    # Retrieve poses...
    try:
        PosePDBQTOutputStr = VinaHandle.poses(
            n_poses=OptionsInfo["NumPoses"], energy_range=OptionsInfo["EnergyRange"], coordinates_only=False
        )
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to retrieve docked poses:\n%s\n" % ErrMsg)
        return (False, None, None, None)

    PoseMol, PoseMolFlexRes = SetupPoseMols(PosePDBQTOutputStr)
    if PoseMol is None:
        return (False, None, None, None)

    # Retrieve  energies...
    try:
        Energies = VinaHandle.energies(n_poses=OptionsInfo["NumPoses"], energy_range=OptionsInfo["EnergyRange"])
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to retrieve energies for docked poses:\n%s\n" % ErrMsg)
        return (False, None, None, None)

    if len(Energies) == 0:
        return (False, None, None, None)

    Status = True

    return (Status, PoseMol, Energies, PoseMolFlexRes)


def SetupPoseMols(PosePDBQTOutputStr):
    """Process PDBQT Vina poses string to setup pose mols for the docked
    molecule and any flexible residues."""

    PoseMol, PoseMolFlexRes = [None, None]

    # Setup a mol containing poses as conformers...
    try:
        PosePDBQTOutputMol = PDBQTMolecule(PosePDBQTOutputStr)
        PoseMols = RDKitMolCreate.from_pdbqt_mol(PosePDBQTOutputMol)
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to retrieve docked poses:\n%s\n" % ErrMsg)
        return (None, None)

    if len(PoseMols) == 0:
        return (None, None)

    # First mol is the pose for docked molecule...
    PoseMol = PoseMols.pop(0)

    # Collect pose mols foe flexible side chain residues...
    PoseMolsFlexRes = []
    for Mol in PoseMols:
        if not Mol.HasProp("meeko"):
            continue
        MeekoPropMap = json.loads(Mol.GetProp("meeko"))
        if type(MeekoPropMap) is dict:
            if "is_sidechain" in MeekoPropMap:
                if MeekoPropMap["is_sidechain"]:
                    PoseMolsFlexRes.append(Mol)

    # Combine pose mols for flexible side chain residues into a single pose mol...
    if len(PoseMolsFlexRes):
        for Mol in PoseMolsFlexRes:
            if PoseMolFlexRes is None:
                PoseMolFlexRes = Mol
            else:
                PoseMolFlexRes = Chem.CombineMols(PoseMolFlexRes, Mol)

    if PoseMolFlexRes is not None:
        PoseMolConfCount = PoseMol.GetNumConformers()
        PoseMolFlexResConfCount = PoseMolFlexRes.GetNumConformers()
        if PoseMolConfCount != PoseMolFlexResConfCount:
            if not OptionsInfo["QuietMode"]:
                MiscUtil.PrintWarning(
                    "The number of poses, %s, for flexible residues doesn't match number of poses, %s, for docked molecule...\n"
                    % (PoseMolFlexResConfCount, PoseMolConfCount)
                )

    return (PoseMol, PoseMolFlexRes)


def PrepareMolecule(MolPrepHandle, Mol):
    """Prepare molecule for docking."""

    try:
        PreppedMols = MolPrepHandle.prepare(Mol)
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to prepare molecule for docking:\n%s\n" % ErrMsg)
        return None

    if len(PreppedMols) == 0:
        return None

    PreppedMol = PreppedMols[0]

    # Setup PDBQT mole string...
    try:
        PDBQTMolStr, Status, ErrMsg = PDBQTWriterLegacy.write_string(PreppedMol)
    except Exception as ExceptionErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to prepare molecule for docking:\n%s\n" % ExceptionErrMsg)
        return None

    if not Status:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to prepare molecule for docking:\n%s\n" % ErrMsg)
        return None

    if MiscUtil.IsEmpty(PDBQTMolStr):
        return None

    return PDBQTMolStr


def InitializeVina(Quiet=False):
    """Initialize AutoDock Vina."""

    if not Quiet:
        MiscUtil.PrintInfo("\nInitializing Vina...")

    VinaHandle = Vina(
        sf_name=OptionsInfo["Forcefield"],
        cpu=OptionsInfo["NumThreads"],
        seed=OptionsInfo["RandomSeed"],
        no_refine=OptionsInfo["SkipRefinement"],
        verbosity=OptionsInfo["VinaVerbosity"],
    )

    SetupReceptor(VinaHandle, Quiet)
    SetupForcefieldWeights(VinaHandle, Quiet)
    SetupReceptorMaps(VinaHandle, Quiet)

    return VinaHandle


def SetupReceptor(VinaHandle, Quiet=False):
    """Setup receptor"""

    if OptionsInfo["UseReceptorFile"] and OptionsInfo["UseReceptorFlexFile"]:
        VinaHandle.set_receptor(
            rigid_pdbqt_filename=OptionsInfo["ReceptorFile"], flex_pdbqt_filename=OptionsInfo["ReceptorFlexFile"]
        )
    elif OptionsInfo["UseReceptorFile"]:
        VinaHandle.set_receptor(rigid_pdbqt_filename=OptionsInfo["ReceptorFile"], flex_pdbqt_filename=None)
    elif OptionsInfo["UseReceptorFlexFile"]:
        VinaHandle.set_receptor(rigid_pdbqt_filename=None, flex_pdbqt_filename=OptionsInfo["ReceptorFlexFile"])


def SetupForcefieldWeights(VinaHandle, Quiet=False):
    """Setup forcefield weights."""

    Weights = OptionsInfo["ForcefieldWeightParams"]
    Forcefield = OptionsInfo["Forcefield"]
    if not OptionsInfo["ForcefieldWeightParamsSpecified"]:
        if not Quiet:
            MiscUtil.PrintInfo("\nUsing default forcefield weights for %s..." % Forcefield)
        return

    if not Quiet:
        MiscUtil.PrintInfo("\nSetting specified forcefield weights for %s..." % Forcefield)

    if OptionsInfo["UseAD4Forcefield"]:
        VinaHandle.set_weights(
            [
                Weights["AD4Vdw"],
                Weights["AD4HydrogenBond"],
                Weights["AD4Electrostatic"],
                Weights["AD4Desolvation"],
                Weights["AD4GlueLinearAttraction"],
                Weights["AD4Rot"],
            ]
        )
    elif OptionsInfo["UseVinaForcefield"]:
        VinaHandle.set_weights(
            [
                Weights["VinaGaussian1"],
                Weights["VinaGaussian2"],
                Weights["VinaRepulsion"],
                Weights["VinaHydrophobic"],
                Weights["VinaHydrogenBond"],
                Weights["VinaGlueLinearAttraction"],
                Weights["VinaRot"],
            ]
        )
    elif OptionsInfo["UseVinardoForcefield"]:
        VinaHandle.set_weights(
            [
                Weights["VinardoGaussian1"],
                Weights["VinardoRepulsion"],
                Weights["VinardoHydrophobic"],
                Weights["VinardoHydrogenBond"],
                Weights["VinardoGlueLinearAttraction"],
                Weights["VinardoRot"],
            ]
        )
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "-f, --forcefield" is not valid. Supported values: AD4 Vina Vinardo'
            % Forcefield
        )


def SetupReceptorMaps(VinaHandle, Quiet=False):
    """Setup receptor maps."""

    if not Quiet:
        MiscUtil.PrintInfo("\nSetting up receptor and maps for %s forcefield..." % OptionsInfo["Forcefield"])

    if OptionsInfo["UseAD4Forcefield"]:
        # Load maps for rigid portion of the receptor...
        VinaHandle.load_maps(OptionsInfo["ReceptorMapsPrefix"])
    elif OptionsInfo["UseVinaForcefield"] or OptionsInfo["UseVinardoForcefield"]:
        # Load or compute maps for rigid portion of the receptor...
        if OptionsInfo["UseReceptorMapsPrefix"]:
            VinaHandle.load_maps(OptionsInfo["ReceptorMapsPrefix"])
        else:
            VinaHandle.compute_vina_maps(
                center=OptionsInfo["GridCenterList"],
                box_size=OptionsInfo["GridSizeList"],
                spacing=OptionsInfo["GridSpacing"],
                force_even_voxels=False,
            )
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "-f, --forcefield" is not valid. Supported values: AD4 Vina Vinardo'
            % OptionsInfo["Forcefield"]
        )


def InitializeMeekoMolPrepration(Quiet=False):
    """Initialize meeko molecule prepration."""

    if OptionsInfo["MergeHydrogens"]:
        MolPrep = MoleculePreparation(merge_these_atom_types=("H",))
    else:
        MolPrep = MoleculePreparation(merge_these_atom_types="")

    return MolPrep


def CheckAndValidateMolecule(Mol, MolCount=None):
    """Validate molecule for docking."""

    MolName = RDKitUtil.GetMolName(Mol, MolCount)
    if RDKitUtil.IsMolEmpty(Mol):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Ignoring empty molecule: %s\n" % MolName)
        return False

    if not OptionsInfo["ValidateMolecules"]:
        return True

    # Check for 3D flag...
    if not Mol.GetConformer().Is3D():
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("3D tag is not set for molecule: %s\n" % MolName)

    # Check for missing hydrogens...
    if RDKitUtil.AreHydrogensMissingInMolecule(Mol):
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Missing hydrogens in molecule: %s\n" % MolName)

    return True


def WriteMolPoses(Writer, Mol, MolNum, PoseMol, Energies, WriterFlexRes=None, PoseMolFlexRes=None):
    """Write molecule."""

    if OptionsInfo["ScoreOnlyMode"]:
        return WriteMolPoseForScoreOnlyMode(Writer, Mol, MolNum, PoseMol, Energies)
    elif OptionsInfo["LocalOptimizationOnlyMode"]:
        return WriteMolPoseForLocalOptimizationOnlyMode(Writer, Mol, MolNum, PoseMol, Energies)
    elif OptionsInfo["DockMode"]:
        return WriteMolPosesForDockMode(Writer, Mol, MolNum, PoseMol, Energies, WriterFlexRes, PoseMolFlexRes)
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "-m, --mode" is not valid. Supported values: Dock LocalOptimizationOnly ScoreOnly'
            % OptionsInfo["Mode"]
        )


def WriteMolPoseForScoreOnlyMode(Writer, Mol, MolNum, PoseMol, Energies):
    """Write out molecule and associated information for score only mode."""

    ClearMeekoMolProperties(PoseMol)

    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    PoseMol.SetProp("_Name", MolName)

    SetupEnergyProperties(PoseMol, Energies)
    Writer.write(PoseMol)

    return


def WriteMolPoseForLocalOptimizationOnlyMode(Writer, Mol, MolNum, PoseMol, Energies):
    """Write out molecule and associated information for score only mode."""

    ClearMeekoMolProperties(PoseMol)

    MolName = RDKitUtil.GetMolName(Mol, MolNum)
    PoseMol.SetProp("_Name", MolName)

    SetupEnergyProperties(PoseMol, Energies)
    Writer.write(PoseMol)


def WriteMolPosesForDockMode(Writer, Mol, MolNum, PoseMol, Energies, WriterFlexRes=None, PoseMolFlexRes=None):
    """Write out molecule and associated information for dock mode."""

    MolName = RDKitUtil.GetMolName(Mol, MolNum)

    # Write out poses for docked molecule...
    ClearMeekoMolProperties(PoseMol)
    for PoseMolConfIndex, PoseMolConf in enumerate(PoseMol.GetConformers()):
        PoseMolName = "%s_Pose%s" % (MolName, (PoseMolConfIndex + 1))
        PoseMol.SetProp("_Name", PoseMolName)

        SetupEnergyProperties(PoseMol, Energies[PoseMolConfIndex])

        Writer.write(PoseMol, confId=PoseMolConf.GetId())

    # Write out poses for flexible reside side chains...
    if WriterFlexRes is None or PoseMolFlexRes is None:
        return

    ClearMeekoMolProperties(PoseMolFlexRes)
    for PoseMolFlexResConfIndex, PoseMolFlexResConf in enumerate(PoseMolFlexRes.GetConformers()):
        PoseMolFlexResName = "%s_Flex_Receptor_Pose%s" % (MolName, (PoseMolFlexResConfIndex + 1))
        PoseMolFlexRes.SetProp("_Name", PoseMolFlexResName)

        WriterFlexRes.write(PoseMolFlexRes, confId=PoseMolFlexResConf.GetId())


def ClearMeekoMolProperties(Mol):
    """Clear Meeko molecule properties."""

    for PropName in ["meeko"]:
        if Mol.HasProp(PropName):
            Mol.ClearProp(PropName)


def SetupEnergyProperties(Mol, Energies):
    """Setup energy properties."""

    Precision = OptionsInfo["Precision"]

    for Index, EnergyLabel in enumerate(OptionsInfo["EnergyLabelsList"]):
        EnergyValueIndex = OptionsInfo["EnergyValueIndicesList"][Index]
        EnergyValue = "%.*f" % (Precision, Energies[EnergyValueIndex])
        Mol.SetProp(EnergyLabel, EnergyValue)


def SetupWriters():
    """Setup writers for output files."""

    Writer, WriterFlexRes = [None, None]

    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])

    if OptionsInfo["DockMode"] and OptionsInfo["UseReceptorFlexFile"]:
        WriterFlexRes = RDKitUtil.MoleculesWriter(OptionsInfo["OutfileFlexRes"], **OptionsInfo["OutfileParams"])
        if WriterFlexRes is None:
            MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["OutfileFlexRes"])

    return (Writer, WriterFlexRes)


def CloseWriters(Writer, WriterFlexRes):
    """Close writers."""

    if Writer is not None:
        Writer.close()

    if WriterFlexRes is not None:
        WriterFlexRes.close()


def ComputeGridCenter(LigandFile):
    """Compute grid center from ligand file."""

    GridCenter = []

    MiscUtil.PrintInfo("\nComputing grid center from reference ligand file %s..." % LigandFile)

    Mols = RDKitUtil.ReadMolecules(LigandFile)
    Mol = Mols[0]

    Centroid = rdMolTransforms.ComputeCentroid(Mol.GetConformer())

    GridCenter = [Centroid.x, Centroid.y, Centroid.z]

    GridCenterFormatted = ["%.3f" % Value for Value in GridCenter]
    MiscUtil.PrintInfo("GridCenter: %s" % (" ".join(GridCenterFormatted)))

    return GridCenter


def ProcessReceptorOptions():
    """Process receptor options."""

    ReceptorFile, ReceptorMapsPrefix, UseReceptorFile, UseReceptorMapsPrefix = [None, None, False, False]

    Receptor = Options["--receptor"]
    if os.path.isfile(Receptor):
        ReceptorFile = Receptor
        UseReceptorFile = True
    else:
        ReceptorMapsPrefix = Receptor
        UseReceptorMapsPrefix = True

    OptionsInfo["Receptor"] = Receptor

    OptionsInfo["ReceptorFile"] = ReceptorFile
    OptionsInfo["UseReceptorFile"] = UseReceptorFile

    OptionsInfo["ReceptorMapsPrefix"] = ReceptorMapsPrefix
    OptionsInfo["UseReceptorMapsPrefix"] = UseReceptorMapsPrefix

    UseReceptorFlexFile = False
    ReceptorFlexFile = None
    if not re.match("^None$", Options["--receptorFlexFile"], re.I):
        UseReceptorFlexFile = True
        ReceptorFlexFile = Options["--receptorFlexFile"]
    OptionsInfo["ReceptorFlexFile"] = ReceptorFlexFile
    OptionsInfo["UseReceptorFlexFile"] = UseReceptorFlexFile

    if OptionsInfo["UseAD4Forcefield"]:
        if OptionsInfo["UseReceptorFile"]:
            MiscUtil.PrintError(
                'The value specified, %s, for option "-r, --receptor" is not valid for %s forcefield. Supported value: Affinity maps prefix.'
                % (OptionsInfo["Receptor"], Options["--forcefield"])
            )


def ProcessForcefieldWeightParamatersOption(ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo=None):
    """Process forcefield weight paramaters option."""

    ParamsInfo = {
        "AD4Vdw": 0.1662,
        "AD4HydrogenBond": 0.1209,
        "AD4Electrostatic": 0.1406,
        "AD4Desolvation": 0.1322,
        "AD4GlueLinearAttraction": 50.0,
        "AD4Rot": 0.2983,
        "VinaGaussian1": -0.035579,
        "VinaGaussian2": -0.005156,
        "VinaRepulsion": 0.840245,
        "VinaHydrophobic": -0.035069,
        "VinaHydrogenBond": -0.587439,
        "VinaGlueLinearAttraction": 50.0,
        "VinaRot": 0.05846,
        "VinardoGaussian1": -0.045,
        "VinardoRepulsion": 0.8,
        "VinardoHydrophobic": -0.035,
        "VinardoHydrogenBond": -0.600,
        "VinardoGlueLinearAttraction": 50.0,
        "VinardoRot": 0.05846,
    }

    # Setup a canonical paramater names...
    ValidParamNames = []
    CanonicalParamNamesMap = {}
    for ParamName in sorted(ParamsInfo):
        ValidParamNames.append(ParamName)
        CanonicalParamNamesMap[ParamName.lower()] = ParamName

    # Update default values...
    if ParamsDefaultInfo is not None:
        for ParamName in ParamsDefaultInfo:
            if ParamName not in ParamsInfo:
                MiscUtil.PrintError(
                    'The default parameter name, %s, specified using "%s" to function ProcessForcefieldWeightParamatersOption is not a valid name. Supported parameter names: %s'
                    % (ParamName, ParamsDefaultInfo, " ".join(ValidParamNames))
                )
            ParamsInfo[ParamName] = ParamsDefaultInfo[ParamName]

    if re.match("^auto$", ParamsOptionValue, re.I):
        return ParamsInfo

    ParamsOptionValueWords = ParamsOptionValue.split(",")
    if len(ParamsOptionValueWords) % 2:
        MiscUtil.PrintError(
            'The number of comma delimited paramater names and values, %d, specified using "%s" option must be an even number.'
            % (len(ParamsOptionValueWords), ParamsOptionName)
        )

    # Validate paramater name and value pairs...
    for Index in range(0, len(ParamsOptionValueWords), 2):
        Name = ParamsOptionValueWords[Index].strip()
        Value = ParamsOptionValueWords[Index + 1].strip()

        CanonicalName = Name.lower()
        if CanonicalName not in CanonicalParamNamesMap:
            MiscUtil.PrintError(
                'The parameter name, %s, specified using "%s" is not a valid name. Supported parameter names: %s'
                % (Name, ParamsOptionName, " ".join(ValidParamNames))
            )

        ParamName = CanonicalParamNamesMap[CanonicalName]

        if not MiscUtil.IsFloat(Value):
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" must be a float.'
                % (Value, Name, ParamsOptionName)
            )
        ParamValue = float(Value)

        ParamsInfo[ParamName] = ParamValue

    return ParamsInfo


def ProcessModeOption():
    """Process mode option."""

    Mode = Options["--mode"]
    DockMode, LocalOptimizationOnlyMode, ScoreOnlyMode = [False] * 3
    if re.match("^Dock$", Mode, re.I):
        Mode = "Dock"
        DockMode = True
    elif re.match("^LocalOptimizationOnly$", Mode, re.I):
        Mode = "LocalOptimizationOnly"
        LocalOptimizationOnlyMode = True
    elif re.match("^ScoreOnly$", Mode, re.I):
        Mode = "ScoreOnly"
        ScoreOnlyMode = True
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "-m, --mode" is not valid. Supported values: Dock LocalOptimizationOnly ScoreOnly'
            % OptionsInfo["Mode"]
        )

    OptionsInfo["Mode"] = Mode
    OptionsInfo["DockMode"] = DockMode
    OptionsInfo["LocalOptimizationOnlyMode"] = LocalOptimizationOnlyMode
    OptionsInfo["ScoreOnlyMode"] = ScoreOnlyMode


def ProcessEnergyLabelOptions():
    """Process energy label options."""

    Forcefield = OptionsInfo["Forcefield"]

    EnergyLabel = Options["--energyLabel"]
    if re.match("^auto$", EnergyLabel, re.I):
        EnergyLabel = "%s_Total_Energy (kcal/mol)" % Forcefield
    OptionsInfo["EnergyLabel"] = EnergyLabel

    EnergyLabelsList = []
    DefaultLabels = [
        "%s_Intermolecular_Energy (kcal/mol)" % (Forcefield),
        "%s_Internal_Energy (kcal/mol)" % (Forcefield),
        "%s_Torsions_Energy (kcal/mol)" % (Forcefield),
    ]

    EnergyLabels = Options["--energyComponentsLabels"]
    if not re.match("^auto$", EnergyLabels, re.I):
        EnergyLabelsWords = EnergyLabels.split(",")
        if len(EnergyLabelsWords) != 3:
            MiscUtil.PrintError(
                'The specified value, %s, for option "--energyComponentsLabels " is not valid. It must contain 3 text values separated by comma.'
                % EnergyLabels
            )

        for LabelIndex, Label in enumerate(EnergyLabelsWords):
            Label = Label.strip()
            if re.match("^auto$", Label, re.I):
                Label = DefaultLabels[LabelIndex]

            EnergyLabelsList.append(Label)
    else:
        EnergyLabelsList = DefaultLabels

    OptionsInfo["EnergyComponentsLabels"] = EnergyLabels
    OptionsInfo["EnergyComponentsLabelsList"] = EnergyLabelsList

    SetupEnergyLabelsAndIndices()


def SetupEnergyLabelsAndIndices():
    """Setup energy data labels and indices for writing out energy values."""

    EnergyLabelsList = []
    EnergyValueIndicesList = []

    # Total energy is always at index 0 in the list retured by vina.score (),
    # vina.optimize(), and vina.energies() during ScoreOnly, LocalOptimizationOnly
    # and Dock modes...
    EnergyLabelsList.append(OptionsInfo["EnergyLabel"])
    EnergyValueIndicesList.append(0)

    OptionsInfo["EnergyLabelsList"] = EnergyLabelsList
    OptionsInfo["EnergyValueIndicesList"] = EnergyValueIndicesList

    if not OptionsInfo["EnergyComponents"]:
        return

    # Setup energy labels and indices for energy components...
    if OptionsInfo["ScoreOnlyMode"]:
        # vina.score returns a list containing the following values:
        #
        # Vina/Vinardo FF: columns=[total, lig_inter, flex_inter, other_inter, flex_intra, lig_intra, torsions, lig_intra best pose]
        # AutoDock FF: [total, lig_inter, flex_inter, other_inter, flex_intra, lig_intra, torsions, -lig_intra]
        #
        IntermolecularEnergyIndex, IntramolecularEnergyIndex, TorsionEnergyIndex = [1, 5, 6]
    elif OptionsInfo["LocalOptimizationOnlyMode"]:
        # vina.optimize returns a list containing the following values:
        #
        #  Vina/Vinardo FF: [total, lig_inter, flex_inter, other_inter, flex_intra, lig_intra, torsions, lig_intra best pose]
        # AutoDock FF: [total, lig_inter, flex_inter, other_inter, flex_intra, lig_intra, torsions, -lig_intra]
        #
        IntermolecularEnergyIndex, IntramolecularEnergyIndex, TorsionEnergyIndex = [1, 5, 6]
    elif OptionsInfo["DockMode"]:
        # vina.energies returns a list containing the following values:
        #
        #  Vina/Vinardo FF: [total, inter, intra, torsions, intra best pose]
        # AutoDock FF: [total, inter, intra, torsions, -intra]
        #
        IntermolecularEnergyIndex, IntramolecularEnergyIndex, TorsionEnergyIndex = [1, 2, 3]
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "-m, --mode" is not valid. Supported values: Dock LocalOptimizationOnly ScoreOnly'
            % OptionsInfo["Mode"]
        )

    OptionsInfo["EnergyLabelsList"].extend(OptionsInfo["EnergyComponentsLabelsList"])
    OptionsInfo["EnergyValueIndicesList"].extend(
        [IntermolecularEnergyIndex, IntramolecularEnergyIndex, TorsionEnergyIndex]
    )


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]
    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters(
        "--infileParams",
        Options["--infileParams"],
        InfileName=Options["--infile"],
        ParamsDefaultInfo=ParamsDefaultInfoOverride,
    )

    OptionsInfo["Outfile"] = Options["--outfile"]
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters(
        "--outfileParams", Options["--outfileParams"]
    )

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    OptionsInfo["OutfileFlexRes"] = "%s_Flex_Receptor.%s" % (FileName, FileExt)

    GridCenterLigandFile, GridCenterList, GridCenterByLigandFile = [None, None, False]
    GridCenter = Options["--gridCenter"]
    if re.search(",", GridCenter, re.I):
        GridCenterList = [float(Value) for Value in GridCenter.split(",")]
    else:
        GridCenterByLigandFile = True
        GridCenterLigandFile = GridCenter
        GridCenterList = ComputeGridCenter(GridCenterLigandFile)

    OptionsInfo["GridCenter"] = GridCenter
    OptionsInfo["GridCenterList"] = GridCenterList
    OptionsInfo["GridCenterLigandFile"] = GridCenterLigandFile
    OptionsInfo["GridCenterByLigandFile"] = GridCenterByLigandFile

    OptionsInfo["EnergyComponents"] = True if re.match("^yes$", Options["--energyComponents"], re.I) else False

    OptionsInfo["EnergyRange"] = float(Options["--energyRange"])
    OptionsInfo["Exhaustiveness"] = int(Options["--exhaustiveness"])

    Forcefield = Options["--forcefield"]
    UseAD4Forcefield, UseVinaForcefield, UseVinardoForcefield = [False, False, False]
    if re.match("^AD4$", Forcefield, re.I):
        Forcefield = "AD4"
        UseAD4Forcefield = True
    elif re.match("^Vina$", Forcefield, re.I):
        Forcefield = "Vina"
        UseVinaForcefield = True
    elif re.match("^Vinardo$", Forcefield, re.I):
        Forcefield = "Vinardo"
        UseVinardoForcefield = True
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "-f, --forcefield" is not valid. Supported values: AD4 Vina Vinardo'
        )
    OptionsInfo["Forcefield"] = Forcefield
    OptionsInfo["UseAD4Forcefield"] = UseAD4Forcefield
    OptionsInfo["UseVinaForcefield"] = UseVinaForcefield
    OptionsInfo["UseVinardoForcefield"] = UseVinardoForcefield

    OptionsInfo["ForcefieldWeightParams"] = ProcessForcefieldWeightParamatersOption(
        "--forcefieldWeightParams", Options["--forcefieldWeightParams"]
    )
    OptionsInfo["ForcefieldWeightParamsSpecified"] = (
        False if re.match("^auto$", Options["--forcefieldWeightParams"], re.I) else True
    )

    GridSize = Options["--gridSize"]
    GridSizeList = [float(Value) for Value in GridSize.split(",")]
    OptionsInfo["GridSize"] = GridSize
    OptionsInfo["GridSizeList"] = GridSizeList

    OptionsInfo["GridSpacing"] = float(Options["--gridSpacing"])

    OptionsInfo["MaxEvaluations"] = int(Options["--maxEvaluations"])
    OptionsInfo["MinRMSD"] = float(Options["--minRMSD"])

    MergeHydrogens = Options["--mergeHydrogens"]
    if re.match("^auto$", MergeHydrogens, re.I):
        MergeHydrogens = True if OptionsInfo["UseAD4Forcefield"] else False
    else:
        MergeHydrogens = True if re.match("^yes$", MergeHydrogens, re.I) else False
    OptionsInfo["MergeHydrogens"] = MergeHydrogens

    ProcessModeOption()

    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])

    OptionsInfo["NumPoses"] = int(Options["--numPoses"])

    if re.match("^auto$", Options["--numThreads"], re.I):
        NumThreads = 1 if OptionsInfo["MPMode"] else 0
    else:
        NumThreads = int(Options["--numThreads"])
    OptionsInfo["NumThreads"] = NumThreads

    OptionsInfo["Precision"] = int(Options["--precision"])
    OptionsInfo["RandomSeed"] = int(Options["--randomSeed"])

    OptionsInfo["SkipRefinement"] = True if re.match("^yes$", Options["--skipRefinement"], re.I) else False

    OptionsInfo["ValidateMolecules"] = True if re.match("^yes$", Options["--validateMolecules"], re.I) else False

    if re.match("^auto$", Options["--vinaVerbosity"], re.I):
        VinaVerbosity = 0 if OptionsInfo["MPMode"] else 1
    else:
        VinaVerbosity = int(Options["--vinaVerbosity"])
    OptionsInfo["VinaVerbosity"] = VinaVerbosity

    OptionsInfo["Overwrite"] = Options["--overwrite"]
    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False

    ProcessEnergyLabelOptions()
    ProcessReceptorOptions()


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
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol")

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    MiscUtil.ValidateOptionsOutputFileOverwrite(
        "-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"]
    )
    MiscUtil.ValidateOptionsDistinctFileNames(
        "-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"]
    )

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--receptor"])
    if not MiscUtil.IsEmpty(FileExt):
        MiscUtil.ValidateOptionFilePath("-r, --receptor", Options["--receptor"])
        MiscUtil.ValidateOptionFileExt("-r, --receptor", Options["--receptor"], "pdbqt")
    else:
        AffinityMapFiles = glob.glob("%s.*.map" % Options["--receptor"])
        if len(AffinityMapFiles) == 0:
            MiscUtil.PrintError(
                'The receptor affinity map files, %s.*.map, corresponding to maps prefix, %s, specified using  option, "-r, --receptor" option don\'t exist.'
                % (Options["--receptor"], Options["--receptor"])
            )

    if not re.match("^None$", Options["--receptorFlexFile"], re.I):
        MiscUtil.ValidateOptionFilePath("--receptorFlexFile", Options["--receptorFlexFile"])
        MiscUtil.ValidateOptionFileExt("--receptorFlexFile", Options["--receptorFlexFile"], "pdbqt")
        if os.path.isfile(Options["--receptor"]):
            MiscUtil.ValidateOptionsDistinctFileNames(
                "-r, --receptor", Options["--receptor"], "--receptorFlexFile", Options["--receptorFlexFile"]
            )

    if re.search(",", Options["--gridCenter"], re.I):
        MiscUtil.ValidateOptionNumberValues("-g, --gridCenter", Options["--gridCenter"], 3, ",", "float", {">=": 0.0})
    else:
        MiscUtil.ValidateOptionFilePath("-g, --gridCenter", Options["--gridCenter"])
        MiscUtil.ValidateOptionFileExt("-g, --gridCenter", Options["--gridCenter"], "sdf sd mol pdb")

    MiscUtil.ValidateOptionTextValue("--energyComponents", Options["--energyComponents"], "yes no")

    MiscUtil.ValidateOptionFloatValue("--energyRange", Options["--energyRange"], {">": 0})

    MiscUtil.ValidateOptionIntegerValue("--exhaustiveness", Options["--exhaustiveness"], {">": 0})

    MiscUtil.ValidateOptionTextValue("-f, --forcefield", Options["--forcefield"], "AD4 Vina Vinardo")

    MiscUtil.ValidateOptionNumberValues("--gridSize", Options["--gridSize"], 3, ",", "float", {">": 0.0})
    MiscUtil.ValidateOptionFloatValue("--gridSpacing", Options["--gridSpacing"], {">": 0})

    MiscUtil.ValidateOptionIntegerValue("--maxEvaluations", Options["--maxEvaluations"], {">=": 0})
    MiscUtil.ValidateOptionFloatValue("--minRMSD", Options["--minRMSD"], {">": 0})

    MiscUtil.ValidateOptionTextValue("--mergeHydrogens", Options["--mergeHydrogens"], "yes no auto")

    MiscUtil.ValidateOptionTextValue("-m, --mode", Options["--mode"], "Dock LocalOptimizationOnly ScoreOnly")

    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")

    MiscUtil.ValidateOptionIntegerValue("--numPoses", Options["--numPoses"], {">": 0})

    if not re.match("^auto$", Options["--numThreads"], re.I):
        MiscUtil.ValidateOptionIntegerValue("--numThreads", Options["--numThreads"], {">": 0})

    MiscUtil.ValidateOptionIntegerValue("-p, --precision", Options["--precision"], {">": 0})
    MiscUtil.ValidateOptionIntegerValue("--randomSeed", Options["--randomSeed"], {})

    MiscUtil.ValidateOptionTextValue("--skipRefinement", Options["--skipRefinement"], "yes no")

    MiscUtil.ValidateOptionTextValue("-v, --validateMolecules", Options["--validateMolecules"], "yes no")

    if not re.match("^auto$", Options["--vinaVerbosity"], re.I):
        MiscUtil.ValidateOptionIntegerValue("--vinaVerbosity", Options["--vinaVerbosity"], {">=": 0})
        MiscUtil.ValidateOptionTextValue("--vinaVerbosity", Options["--vinaVerbosity"], "0 1 2")

    MiscUtil.ValidateOptionTextValue("-q, --quiet", Options["--quiet"], "yes no")


# Setup a usage string for docopt...
_docoptUsage_ = """
VinaPerformDocking.py - Perform docking

Usage:
    VinaPerformDocking.py [--energyComponents <yes or no>] [--energyComponentsLabels <Label1, label2, Label3>]
                                  [--energyLabel <text>] [--energyRange <number>] [--exhaustiveness <number>]
                                  [--forcefield <AD4, Vina, or Vinardo>] [--forcefieldWeightParams <Name,Value,...>] [--gridSpacing <number>]
                                  [--gridSize <xsize,ysize,zsize>] [--infileParams <Name,Value,...>] [--maxEvaluations <number>]
                                  [--mergeHydrogens <yes or no>] [--minRMSD <number>] [--mode <Dock, LocalOptimizationOnly, ScoreOnly>] [--mp <yes or no>]
                                  [--mpParams <Name, Value,...>] [--numPoses <number>] [--numThreads <number>] [--outfileParams <Name,Value,...> ]
                                  [--overwrite] [--precision <number>] [--quiet <yes or no>] [--randomSeed <number>] [--receptorFlexFile <receptor flex file>]
                                  [--skipRefinement <yes or no>] [--validateMolecules <yes or no>] [--vinaVerbosity <number>]
                                  [-w <dir>] -g <RefLigandFile or x,y,z> -r <receptorfile or maps prerfix> -i <infile> -o <outfile> 
    VinaPerformDocking.py -h | --help | -e | --examples

Description:
    Dock molecules against a protein receptor using AutoDock Vina [Ref 168, 169].
    The molecules must have 3D coordinates in input file. In addition, the hydrogens
    must be present for all molecules in input file.

    No protein receptor preparation is performed during docking. It must be prepared
    employing standalone scripts available as part of AutoDock Vina. You may
    optionally specify flexible residues in the binding pocket to prepare a flexible
    receptor file and employ it for docking molecules along with the fixed receptor
    file.

    The following three forecefileds are available to score molecules: AD4 
    (AutoDock4), Vina, and Vinardo (Vina RaDii Optimized) [Ref 170].

    The supported input file formats are shown below:
        
        Rigid/Flexible protein receptor files  - PDBQT(.pdbqt)
        Reference ligand file: PDB(.pdb), Mol (.mol), SD (.sdf, .sd)
        
        Input molecules file - Mol (.mol), SD (.sdf, .sd)
        
    The supported output file format is: SD (.sdf, .sd).

    The following output files are generated:
        
        <OutfileRoot>.<OutfileExt> - Docked/scored molecules
        <OutfileRoot>_Flex_Receptor.<OutfileExt> - Docked poses for flexible
            residues 
        
    The flexible receptor output file contains docked poses corresponding to
    flexible residues. It is only generated during 'Dock' value of '-m, --mode'
    option. The number of poses in this file matches those written to the
    output file containing docked molecules.

Options:
    --energyComponents <yes or no>  [default: no]
        Write out binding energy components of the total binding energy docking
        score to outfile. The following three energy components are written to
        outfile: intermolecula energy, internal energy, and torsions energy. 
    --energyComponentsLabels <Label1, label2, Label3>  [default: auto]
        A triplet of comma delimited values corresponding to energy data field
        labels for writing out  the binding energy components to outfile. You must
        specify all three values.  A value of 'None' implies the use of the default
        labels as shown below:
            
            Label1: <ForcefieldName>_Intermolecular_Energy (kcal/mol)
            Label2: <ForcefieldName>_Internal_Energy (kcal/mol)
            Label3: <ForcefieldName>_Torsions_Energy (kcal/mol)
            
    --energyLabel <text>  [default: auto]
        Energy data field label for writing out binding energy docking score to
        output file. Default: <ForcefieldName>_Total_Energy (kcal/mol).
    --energyRange <number>  [default: 3.0]
        Maximum energy difference from the best pose during the generation of
        poses. Units: kcal/mol.
    -e, --examples
        Print examples.
    --exhaustiveness <number>  [default: 8]
        Exhaustiveness of global MC search. The higher values make the search
        more exhaustive and it takes longer to complete. You may want to use
        '16' or '32' as the value of '--exhaustiveness ' to increase the accuracy of
        your pose prediction.
    -f, --forcefield <AD4, Vina, or Vinardo>  [default: Vina]
        Forcefield to use for scoring. Possible values: AD4 (AutoDock 4), Vina
        [Ref 169, 169],  or Vinardo (Vina RaDii Optimized) [Ref 170].
        
        You must specify affinity maps using '-r, --receptor' option during the use
        of 'AD4' forcefield.
    --forcefieldWeightParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for forcefield
        scoring.
        
        The supported parameter names along with their default values are
        are shown below for different forcefields:
            
            AD4 (6 weights):
            
            ad4Vdw, 0.1662, ad4HydrogenBond, 0.1209, ad4Electrostatic, 0.1406,
            ad4Desolvation, 0.1322, ad4GlueLinearAttraction, 50.0,
            ad4Rot, 0.2983
        
            Vina (7 weights):
            
            vinaGaussian1, -0.035579, vinaGaussian2, -0.005156,
            vinaRepulsion, 0.840245, vinaHydrophobic, -0.035069,
            vinaHydrogenBond, -0.587439, vinaGlueLinearAttraction, 50.0,
            vinaRot, 0.05846
        
            Vinardo (6 weights):
            
            vinardoGaussian1, -0.045, vinardoRepulsion, 0.8, 
            vinardoHydrophobic, -0.035, vinardoHydrogenBond, -0.600
            vinardoGlueLinearAttraction, 50.0, vinardoRot, 0.05846
            
        The glue weight parameter corresponds to linear attraction for macrocycle
        closure and has the same value for AD4, Vina, and Vinardo. The rot weight
        has the same value for Vina and Vinardo.
    -g, --gridCenter <RefLigandFile or x,y,z>
        Reference ligand file for calculating the docking grid center or a triplet
        of comma delimited values in Angstrom corresponding to grid center.
        
        This is required option. However, it is ignored during the specification
        of maps prefix for '-r, --receptor' option. 
    --gridSize <xsize,ysize,zsize>  [default: 25.0, 25.0, 25.0]
        Docking grid size in Angstrom.
    --gridSpacing <number>  [default: 0.375]
        Docking grid spacing in Angstrom.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name containing molecules for docking against a receptor. The
        molecules must have 3D coordinates in input file. In addition, the hydrogens
        must be present for all molecules in input file. The input file may contain 3D
        conformers.
    --infileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for reading
        molecules from files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD, MOL: removeHydrogens,no,sanitize,yes,strictParsing,yes
            
    --maxEvaluations <number>  [default: 0]
        Maximum number of evaluations to perform for each MC run during docking.
        By default, its value is set 0 and the number of MC evaluations is determined
        using heuristic rules.
    --mergeHydrogens <yes or no>  [default: auto]
        Merge hydrogens during preparation of molecules for docking. The hydrogens
        are automatically merged during 'AD4' value of '-f, --forcefield' option and its
        value is set to 'yes'. Otherwise, it's set to 'no'.
    --minRMSD <number>  [default: 1.0]
        Minimum RMSD between output poses in Angstrom.
    -m, --mode <Dock, LocalOptimizationOnly, ScoreOnly>  [default: Dock]
        Dock molecules or simply score molecules without performing any docking.
        The supported values along with a brief explanation of the expected
        behavior are shown below:
            
            Dock: Global search along with local optimization and scoring after
                docking
            LocalOptimizationOnly: Local optimization and scoring without any docking
            ScoreOnly: Scoring without any local optimizatiom and docking
            
        The 'ScoreOnly" allows you to score 3D moleculed from input file which
        are already positioned in a binding pocket of a receptor.
    --mp <yes or no>  [default: no]
        Use multiprocessing.
         
        By default, input data is retrieved in a lazy manner via mp.Pool.imap()
        function employing lazy RDKit data iterable. This allows processing of
        arbitrary large data sets without any additional requirements memory.
        
        All input data may be optionally loaded into memory by mp.Pool.map()
        before starting worker processes in a process pool by setting the value
        of 'inputDataMode' to 'InMemory' in '--mpParams' option.
        
        A word to the wise: The default 'chunkSize' value of 1 during 'Lazy' input
        data mode may adversely impact the performance. The '--mpParams' section
        provides additional information to tune the value of 'chunkSize'.
    --mpParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs to configure
        multiprocessing.
        
        The supported parameter names along with their default and possible
        values are shown below:
        
            chunkSize, auto
            inputDataMode, Lazy   [ Possible values: InMemory or Lazy ]
            numProcesses, auto   [ Default: mp.cpu_count() ]
        
        These parameters are used by the following functions to configure and
        control the behavior of multiprocessing: mp.Pool(), mp.Pool.map(), and
        mp.Pool.imap().
        
        The chunkSize determines chunks of input data passed to each worker
        process in a process pool by mp.Pool.map() and mp.Pool.imap() functions.
        The default value of chunkSize is dependent on the value of 'inputDataMode'.
        
        The mp.Pool.map() function, invoked during 'InMemory' input data mode,
        automatically converts RDKit data iterable into a list, loads all data into
        memory, and calculates the default chunkSize using the following method
        as shown in its code:
        
            chunkSize, extra = divmod(len(dataIterable), len(numProcesses) * 4)
            if extra: chunkSize += 1
        
        For example, the default chunkSize will be 7 for a pool of 4 worker processes
        and 100 data items.
        
        The mp.Pool.imap() function, invoked during 'Lazy' input data mode, employs
        'lazy' RDKit data iterable to retrieve data as needed, without loading all the
        data into memory. Consequently, the size of input data is not known a priori.
        It's not possible to estimate an optimal value for the chunkSize. The default 
        chunkSize is set to 1.
        
        The default value for the chunkSize during 'Lazy' data mode may adversely
        impact the performance due to the overhead associated with exchanging
        small chunks of data. It is generally a good idea to explicitly set chunkSize to
        a larger value during 'Lazy' input data mode, based on the size of your input
        data and number of processes in the process pool.
        
        The mp.Pool.map() function waits for all worker processes to process all
        the data and return the results. The mp.Pool.imap() function, however,
        returns the the results obtained from worker processes as soon as the
        results become available for specified chunks of data.
        
        The order of data in the results returned by both mp.Pool.map() and 
        mp.Pool.imap() functions always corresponds to the input data.
    -n, --numPoses <number>  [default: 1]
        Number of docked poses to generate for each molecule and write out
        to output file. This option is only valid for 'Dock' value of '-m, --mode'
        option.
    --numThreads <number>  [default: auto]
        Number of threads/CPUs to use for MC search calculation in Vina. The
        default value is set to 1 during multiprocessing for 'Yes' value of '--mp'
        option. Otherwise, it's set to 0 and rely on Vina detect and use all
        available CPUs for multi-threading.
    -o, --outfile <outfile>
        Output file name for writing out molecules. The flexible receptor residues
        are written to <OutfileRoot>_Flex_Receptor.<OutfileExt>.
    --outfileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        molecules to files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: kekulize,yes,forceV3000,no
            
    --overwrite
        Overwrite existing files.
    --precision <number>  [default: 2]
        Floating point precision for writing energy values.
    -q, --quiet <yes or no>  [default: no]
        Use quiet mode. The warning and information messages will not be printed.
    --randomSeed <number>  [default: 0]
        Random seed for MC search calculations. A value of zero implies it's randomly
        chosen by Vina during the calculation.
    -r, --receptor <receptor file or maps prerfix>
        Protein receptor file name or prefix for affinity map files corresponding
        to the fixed portion of the receptor.
        
        You must specify affinity map files for 'AD4' forcefield. The affinity map
        files correspond to <MapsPrefix>.*.map and must be present
        
        The supported receptor file format is PDBQT (.pdbqt). It must contain a
        prepared protein receptor ready for docking. You may prepare a PDBQT
        receptor file from a PDB file employing the command line scripts available
        with AutoDock Vina and Meeko. For example: prepare_receptor,
        prepare_flexreceptor.py, or or mk_make_recepror.py.
        
        You may want to perform the following steps to clean up your PDB file
        before generating a PDBQT receptor file: Remove extraneous molecules
        such as solvents, ions, and ligand etc.; Extract data for a chain containing
        the binding pocket of interest.
    --receptorFlexFile <receptor flex file>  [default: none]
        Protein receptor file name corresponding to the flexible portion of the
        receptor. The supported receptor file format is PDBQT (.pdbqt). It must
        contain a prepared protein receptor ready for docking. You may prepare
        a flexible PDBQT receptor file from a PDB file employing the command line
        script prepare_flexreceptor or mk_make_recepror.py available with Autodock
        Vina and Meeko.
    --skipRefinement <yes or no>  [default: no]
        Skip refinement. Vina is initialized to skip the use of explicit receptor atoms,
        instead of precalculated grids, during the following three docking modes:
        Dock, LocalOptimizationOnly, and ScoreOnly.
    -v, --validateMolecules <yes or no>  [default: yes]
        Validate molecules for docking. The input molecules must have 3D coordinates
        and the hydrogens must be present. You may skip validation of molecules in
        input file containing all valid molecules.
    --vinaVerbosity <number>  [default: auto]
        Verbosity level for running Vina. Possible values: 0 - No output; 1 - Normal
        output; 2 - Verbose. Default: 0 during multiprocessing for 'Yes' value of '--mp'
        option; otherwise, its set to 1. A non-zero value is not recommended during
        multiprocessing. It doesn't work due to the mingling of the Vina output
        from multiple processes.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To dock molecules using Vina forcefield against a prepared receptor
    corresponding to chain A extracted from PDB file 1R4L.pdb, multi-threading
    across all available CPUs during Vina MC search, calculating grid center form
    a reference ligand file, using grid box size of 25 Angstrom, generating one
    pose for each molecule, and write out a SD file containing docked molecules,
    type:

        % VinaPerformDocking.py -g SampleACE2RefLigand.pdb
          -r SampleACE2Receptor.pdbqt -i SampleACE2Ligands.sdf
          -o SampleACE2LigandsOut.sdf

    To run the first example for generating multiple docked poses for each
    molecule and write out all energy terms to a SD file, type:

        % VinaPerformDocking.py -g SampleACE2RefLigand.pdb
          -r SampleACE2Receptor.pdbqt  --numPoses 5 --energyComponents yes
          -i SampleACE2Ligands.sdf -o SampleACE2LigandsOut.sdf

    To run the first example for docking molecules using Vinardo forcefield and write
    out a SD file, type:

        % VinaPerformDocking.py -f Vinardo -g SampleACE2RefLigand.pdb
          -r SampleACE2Receptor.pdbqt -i SampleACE2Ligands.sdf
          -o SampleACE2LigandsOut.sdf

    To run the first example for docking molecules using AD4 forcefield relying on
    the presence of affinity maps in the working directory and write out a SD file,
    type:

        % VinaPerformDocking.py -f AD4 -g SampleACE2RefLigand.pdb
          -r SampleACE2Receptor -i SampleACE2Ligands.sdf
          -o SampleACE2LigandsOut.sdf

    To run the first example for docking molecules using a set of explicit values
    for grid dimensions and write out a SD file, type:

        % VinaPerformDocking.py -g "41.399, 5.851, 28.082"
          --gridSize "25.0, 25.0, 25.0" --gridSpacing 0.375
          -r SampleACE2Receptor.pdbqt -i SampleACE2Ligands.sdf
          -o SampleACE2LigandsOut.sdf 

    To run the first example for only scoring molecules already positioned in the
    binding pocket and write out a SD file, type:

        % VinaPerformDocking.py -m ScoreOnly -g SampleACE2RefLigand.sdf
          -r SampleACE2Receptor.pdbqt -i SampleACE2RefLigandWithHs.sdf
          -o SampleACE2LigandsOut.sdf

    To run the first example for docking molecules to increase the accuracy of
    pose predictions and write out a SD file, type:

        % VinaPerformDocking.py -g SampleACE2RefLigand.pdb
          -r SampleACE2Receptor.pdbqt --exhaustiveness 24
          -i SampleACE2Ligands.sdf -o SampleACE2LigandsOut.sdf

    To run the first example in multiprocessing mode on all available CPUs
    without loading all data into memory, a single thread for Vina docking, and
    write out a SD file, type:

        % VinaPerformDocking.py -g SampleACE2RefLigand.pdb
          -r SampleACE2Receptor.pdbqt --mp yes
          -i SampleACE2Ligands.sdf -o SampleACE2LigandsOut.sdf

    To run the first example in multiprocessing mode on all available CPUs
    by loading all data into memory, a single thread for Vina, and write out
    a SD file, type:

        % VinaPerformDocking.py -g SampleACE2RefLigand.pdb
          -r SampleACE2Receptor.pdbqt --mp yes --mpParams "inputDataMode,
          InMemory" -i SampleACE2Ligands.sdf -o SampleACE2LigandsOut.sdf

    To run the first example in multiprocessing mode on specific number of CPUs
    and chunk size without loading all data into memory along with a specific number
    of threads for Vina docking and write out a SD file, type:

        % VinaPerformDocking.py -g SampleACE2RefLigand.pdb
          -r SampleACE2Receptor.pdbqt --mp yes --mpParams "inputDataMode,lazy,
          numProcesses,4,chunkSize,2" --numThreads 2
          -i SampleACE2Ligands.sdf -o SampleACE2LigandsOut.sdf

    To run the first example for docking molecules employing a flexible portion
    of the receptor corresponding to ARG273 and write out a SD file, type:

        % VinaPerformDocking.py -g SampleACE2RefLigand.pdb
          -r SampleACE2RigidReceptor.pdbqt
          --receptorFlexFile SampleACE2FlexReceptor.pdbqt
          -i SampleACE2Ligands.sdf -o SampleACE2LigandsOut.sdf

    To run the first example for docking molecules using specified parameters and
    write out a SD file, type:

        % VinaPerformDocking.py -g "41.399, 5.851, 28.082"
          --gridSize "25.0, 25.0, 25.0" --gridSpacing 0.375 --energyComponents
          yes  --exhaustiveness 32 --forcefield Vina --mode dock --numPoses 2
          --numThreads 4 --randomSeed 42 --validateMolecules no
          --vinaVerbosity  0 -r SampleACE2Receptor.pdbqt
          -i SampleACE2Ligands.sdf -o SampleACE2LigandsOut.sdf 

Author:
    Manish Sud(msud@san.rr.com)

Acknowledgments:
    Diogo Santos-Martins and Stefano Forli

See also:
    PyMOLConvertLigandFileFormat.py, PyMOLExtractSelection.py,
    PyMOLInfoMacromolecules.py, PyMOLVisualizeMacromolecules.py,
    RDKitConvertFileFormat.py, RDKitEnumerateTautomers.py,
    RDKitGenerateConformers.py, RDKitPerformMinimization.py,
    RDKitPerformConstrainedMinimization.py

Copyright:
    Copyright (C) 2026 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using AutoDockVina
    and Meeko, open source software packages for docking, and RDKit, an open
    source toolkit for cheminformatics developed by Greg Landrum.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
