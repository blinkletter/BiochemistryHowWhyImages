#!/usr/bin/env python
#
# File: OpenMMPrepareMacromolecule.py
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
    import pdbfixer
    from pdbfixer.pdbfixer import __version__ as pdbfixerversion
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
        "\n%s (OpenMM v%s; PDBFixerVersion v%s; MayaChemTools v%s; %s): Starting...\n"
        % (
            ScriptName,
            mm.Platform.getOpenMMVersion(),
            pdbfixerversion,
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
    PrepareMacromolecule()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def PrepareMacromolecule():
    """Prepare a macromolecule for simulation and write it out."""

    # Read macromolecule...
    PDBFixerObjectHandle = ReadMacromolecule()

    # Identify missing residues...
    IdentifyMissingResidues(PDBFixerObjectHandle)

    # Identify and replace non-standard residues...
    IdentifyAndReplaceNonStandardResidues(PDBFixerObjectHandle)

    # Identify and delete heterogen residues...
    IdentifyAndDeleteHeterogenResidues(PDBFixerObjectHandle)

    # Identify and add missing atoms...
    IdentifyAndAddMissingAtoms(PDBFixerObjectHandle)

    if OptionsInfo["WaterBox"]:
        AddWaterBox(PDBFixerObjectHandle)
    elif OptionsInfo["Membrane"]:
        AddLipidMembrane(PDBFixerObjectHandle)
    else:
        MiscUtil.PrintInfo("\nSkipping addtion of water box or membrane...")

    # Write macromolecule...
    WriteMacromolecule(PDBFixerObjectHandle)


def ReadMacromolecule():
    """Read macromolecule."""

    Infile = OptionsInfo["Infile"]
    MiscUtil.PrintInfo("\nProcessing file %s..." % Infile)

    PDBFixerObjectHandle = pdbfixer.pdbfixer.PDBFixer(filename=Infile)

    return PDBFixerObjectHandle


def WriteMacromolecule(PDBFixerObjectHandle):
    """Write macromolecule."""

    MiscUtil.PrintInfo("\nGenerating output file %s..." % OptionsInfo["Outfile"])
    OpenMMUtil.WritePDBFile(
        OptionsInfo["Outfile"], PDBFixerObjectHandle.topology, PDBFixerObjectHandle.positions, KeepIDs=True
    )


def IdentifyMissingResidues(PDBFixerObjectHandle):
    """Identify missing residues based on PDB REMARK records."""

    MiscUtil.PrintInfo("\nIdentifying missing residues...")

    PDBFixerObjectHandle.missingResidues = {}
    PDBFixerObjectHandle.findMissingResidues()

    ListMissingResidues(PDBFixerObjectHandle)


def ListMissingResidues(PDBFixerObjectHandle):
    """List missing residues."""

    if len(PDBFixerObjectHandle.missingResidues) == 0:
        MiscUtil.PrintInfo("Number of missing residues: 0")
        return

    Chains = list(PDBFixerObjectHandle.topology.chains())

    MissingResiduesCount = 0
    MissingResiduesInfo = []
    for Index, Key in enumerate(sorted(PDBFixerObjectHandle.missingResidues)):
        ChainIndex = Key[0]
        # ResidueIndex corresponds to the residue after the insertion of missing residues...`
        ResidueIndex = Key[1]

        MissingResidues = PDBFixerObjectHandle.missingResidues[Key]
        MissingResiduesCount += len(MissingResidues)

        Chain = Chains[ChainIndex]
        ChainResidues = list(Chain.residues())

        if ResidueIndex < len(ChainResidues):
            ResidueNumOffset = int(ChainResidues[ResidueIndex].id) - len(MissingResidues) - 1
        else:
            ResidueNumOffset = int(ChainResidues[-1].id)

        StartResNum = ResidueNumOffset + 1
        EndResNum = ResidueNumOffset + len(MissingResidues)

        MissingResiduesInfo.append(
            "Chain: %s; StartResNum-EndResNum: %s-%s; Count: %s; Residues: %s"
            % (Chain.id, StartResNum, EndResNum, len(MissingResidues), ", ".join(MissingResidues))
        )

    MiscUtil.PrintInfo("Total number of missing residues: %s" % (MissingResiduesCount))
    if OptionsInfo["ListDetails"]:
        MiscUtil.PrintInfo("%s" % ("\n".join(MissingResiduesInfo)))


def IdentifyAndReplaceNonStandardResidues(PDBFixerObjectHandle):
    """Identify and replace non-standard residues."""

    MiscUtil.PrintInfo("\nIdentifying non-standard residues...")

    PDBFixerObjectHandle.findNonstandardResidues()
    NonStandardResidues = PDBFixerObjectHandle.nonstandardResidues
    NumNonStandardResiues = len(NonStandardResidues)

    MiscUtil.PrintInfo("Total number of non-standard residues: %s" % NumNonStandardResiues)

    if NumNonStandardResiues == 0:
        return

    if OptionsInfo["ListDetails"]:
        ListNonStandardResidues(PDBFixerObjectHandle)

    if not OptionsInfo["ReplaceNonStandardResidues"]:
        MiscUtil.PrintInfo("Skipping replacement of non-standard residues...")
        return

    MiscUtil.PrintInfo("Replacing non-standard residues...")
    PDBFixerObjectHandle.replaceNonstandardResidues()


def ListNonStandardResidues(PDBFixerObjectHandle):
    """List non-standard residues."""

    NonStandardResidues = PDBFixerObjectHandle.nonstandardResidues

    if len(NonStandardResidues) == 0:
        return

    NonStandardResiduesMappingInfo = []
    for Values in NonStandardResidues:
        NonStandardRes = Values[0]
        StandardResName = Values[1]

        MappingInfo = "<%s %s %s>: %s" % (
            NonStandardRes.id,
            NonStandardRes.name,
            NonStandardRes.chain.id,
            StandardResName,
        )
        NonStandardResiduesMappingInfo.append(MappingInfo)

    if len(NonStandardResiduesMappingInfo):
        MiscUtil.PrintInfo("Non-standard residues mapping: %s\n" % ",".join(NonStandardResiduesMappingInfo))


def IdentifyAndDeleteHeterogenResidues(PDBFixerObjectHandle):
    """Identify and delete heterogen residues."""

    DeleteHeterogens = OptionsInfo["DeleteHeterogens"]
    if DeleteHeterogens is None:
        MiscUtil.PrintInfo("\nSkipping deletion of any heterogen residues...")
        return

    if re.match("^All$", DeleteHeterogens, re.I):
        MiscUtil.PrintInfo("\nDeleting all heterogen residues including water...")
        PDBFixerObjectHandle.removeHeterogens(keepWater=False)
    elif re.match("^AllExceptWater$", DeleteHeterogens, re.I):
        MiscUtil.PrintInfo("\nDeleting all heterogen residues except water...")
        PDBFixerObjectHandle.removeHeterogens(keepWater=True)
    elif re.match("^WaterOnly$", DeleteHeterogens, re.I):
        MiscUtil.PrintInfo("\nDeleting water only heterogen residues...")
        DeleteWater(PDBFixerObjectHandle)
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "--deleteHeterogens" is not valid. Supported values: All AllExceptWater WaterOnly None'
            % DeleteHeterogens
        )


def DeleteWater(PDBFixerObjectHandle):
    """Delete water."""

    ModellerObjectHandle = mm.app.Modeller(PDBFixerObjectHandle.topology, PDBFixerObjectHandle.positions)
    ModellerObjectHandle.deleteWater()

    PDBFixerObjectHandle.topology = ModellerObjectHandle.topology
    PDBFixerObjectHandle.positions = ModellerObjectHandle.positions


def IdentifyAndAddMissingAtoms(PDBFixerObjectHandle):
    """Identify and missing atoms along with already identified missing residues."""

    MiscUtil.PrintInfo("\nIdentifying missing atoms...")
    PDBFixerObjectHandle.findMissingAtoms()

    ListMissingAtoms(PDBFixerObjectHandle)

    if OptionsInfo["AddHeavyAtoms"] and OptionsInfo["AddResidues"]:
        Msg = "Adding missing heavy atoms along with missing residues..."
    elif OptionsInfo["AddHeavyAtoms"]:
        Msg = "Adding missing heavy atoms..."
    elif OptionsInfo["AddResidues"]:
        Msg = "Adding missing residues..."
    else:
        Msg = "Skipping addition of any heavy atoms along with any missing residues..."

    if not OptionsInfo["AddHeavyAtoms"]:
        PDBFixerObjectHandle.missingAtoms = {}
        PDBFixerObjectHandle.missingTerminals = {}

    MiscUtil.PrintInfo("\n%s" % Msg)
    if OptionsInfo["AddHeavyAtoms"] or OptionsInfo["AddResidues"]:
        PDBFixerObjectHandle.addMissingAtoms()

    if OptionsInfo["AddHydrogens"]:
        MiscUtil.PrintInfo("Adding missing hydrogens at pH %s..." % OptionsInfo["AddHydrogensAtpH"])
        PDBFixerObjectHandle.addMissingHydrogens(pH=OptionsInfo["AddHydrogensAtpH"], forcefield=None)
    else:
        MiscUtil.PrintInfo("Skipping addition of any missing hydrogens...")


def ListMissingAtoms(PDBFixerObjectHandle):
    """List missing atoms."""

    if len(PDBFixerObjectHandle.missingAtoms) == 0 and len(PDBFixerObjectHandle.missingTerminals) == 0:
        MiscUtil.PrintInfo("Total number of missing atoms: 0")
        return

    ListMissingAtomsInfo(PDBFixerObjectHandle.missingAtoms, TerminalAtoms=False)
    ListMissingAtomsInfo(PDBFixerObjectHandle.missingTerminals, TerminalAtoms=True)


def ListMissingAtomsInfo(MissingAtoms, TerminalAtoms):
    """Setup missing atoms information."""

    MissingAtomsInfo = []
    MissingAtomsCount = 0
    MissingAtomsResiduesCount = 0
    for Residue, Atoms in MissingAtoms.items():
        MissingAtomsResiduesCount += 1
        MissingAtomsCount += len(Atoms)

        if TerminalAtoms:
            AtomsNames = [AtomName for AtomName in Atoms]
        else:
            AtomsNames = [Atom.name for Atom in Atoms]
        AtomsInfo = "<%s %s %s>: %s" % (Residue.id, Residue.name, Residue.chain.id, ", ".join(AtomsNames))
        MissingAtomsInfo.append(AtomsInfo)

    AtomsLabel = "terminal atoms" if TerminalAtoms else "atoms"
    if MissingAtomsCount == 0:
        MiscUtil.PrintInfo("Total number of missing %s: %s" % (MissingAtomsCount, AtomsLabel))
    else:
        ResiduesLabel = "residue" if MissingAtomsResiduesCount == 1 else "residues"
        MiscUtil.PrintInfo(
            "Total number of missing %s across %s %s: %s"
            % (AtomsLabel, MissingAtomsResiduesCount, ResiduesLabel, MissingAtomsCount)
        )
        if OptionsInfo["ListDetails"]:
            MiscUtil.PrintInfo("Missing %s across residues: %s" % (AtomsLabel, "; ".join(MissingAtomsInfo)))


def AddWaterBox(PDBFixerObjectHandle):
    """Add water box."""

    MiscUtil.PrintInfo("\nAdding water box...")

    WaterBoxParams = OptionsInfo["WaterBoxParams"]

    Size, Padding, Shape = [None] * 3
    if WaterBoxParams["ModeSize"]:
        SizeList = WaterBoxParams["SizeList"]
        Size = mm.Vec3(SizeList[0], SizeList[1], SizeList[2]) * mm.unit.nanometer
    elif WaterBoxParams["ModePadding"]:
        Padding = WaterBoxParams["Padding"] * mm.unit.nanometer
        Shape = WaterBoxParams["Shape"]
    else:
        MiscUtil.PrintError(
            'The parameter value, %s, specified for parameter name, mode, using "--waterBoxParams" option is not a valid value. Supported values: Size Padding \n'
            % (WaterBoxParams["Mode"])
        )

    IonicStrength = OptionsInfo["IonicStrength"] * mm.unit.molar
    PDBFixerObjectHandle.addSolvent(
        boxSize=Size,
        padding=Padding,
        boxShape=Shape,
        positiveIon=OptionsInfo["IonPositive"],
        negativeIon=OptionsInfo["IonNegative"],
        ionicStrength=IonicStrength,
    )


def AddLipidMembrane(PDBFixerObjectHandle):
    """Add lipid membrane along with water."""

    MiscUtil.PrintInfo("\nAdding membrane along with water...")

    MembraneParams = OptionsInfo["MembraneParams"]

    LipidType = MembraneParams["LipidType"]
    MembraneCenterZ = MembraneParams["MembraneCenterZ"] * mm.unit.nanometer
    Padding = MembraneParams["Padding"] * mm.unit.nanometer

    IonicStrength = OptionsInfo["IonicStrength"] * mm.unit.molar

    PDBFixerObjectHandle.addMembrane(
        lipidType=LipidType,
        membraneCenterZ=MembraneCenterZ,
        minimumPadding=Padding,
        positiveIon=OptionsInfo["IonPositive"],
        negativeIon=OptionsInfo["IonNegative"],
        ionicStrength=IonicStrength,
    )


def ProcessMembraneParamsOption():
    """Process option for membrane parameters."""

    ParamsOptionName = "--membraneParams"
    ParamsOptionValue = Options[ParamsOptionName]
    ParamsDefaultInfo = {"LipidType": ["str", "POPC"], "MembraneCenterZ": ["float", 0.0], "Padding": ["float", 1.0]}
    MembraneParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    for ParamName in ["LipidType", "MembraneCenterZ", "Padding"]:
        ParamValue = MembraneParams[ParamName]
        if ParamName == "LipidType":
            LipidTypes = ["POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"]
            if ParamValue not in LipidTypes:
                MiscUtil.PrintError(
                    'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: %s \n'
                    % (ParamValue, ParamName, ParamsOptionName, LipidTypes)
                )
        elif ParamName == "Padding":
            if ParamValue <= 0:
                MiscUtil.PrintError(
                    'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                    % (ParamValue, ParamName, ParamsOptionName)
                )

    OptionsInfo["MembraneParams"] = MembraneParams


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["Infile"])
    OptionsInfo["InfileRoot"] = FileName

    OptionsInfo["Outfile"] = Options["--outfile"]

    OptionsInfo["AddHeavyAtoms"] = True if re.match("^yes$", Options["--addHeavyAtoms"], re.I) else False

    OptionsInfo["AddHydrogens"] = True if re.match("^yes$", Options["--addHydrogens"], re.I) else False
    OptionsInfo["AddHydrogensAtpH"] = float(Options["--addHydrogensAtpH"])

    OptionsInfo["AddResidues"] = True if re.match("^yes$", Options["--addResidues"], re.I) else False

    DeleteHeterogens = Options["--deleteHeterogens"]
    if re.match("^All$", DeleteHeterogens, re.I):
        DeleteHeterogens = "All"
    elif re.match("^AllExceptWater$", DeleteHeterogens, re.I):
        DeleteHeterogens = "AllExceptWater"
    elif re.match("^WaterOnly$", DeleteHeterogens, re.I):
        DeleteHeterogens = "WaterOnly"
    elif re.match("^None$", DeleteHeterogens, re.I):
        DeleteHeterogens = None
    elif re.match("^auto$", DeleteHeterogens, re.I):
        DeleteHeterogens = None
        if re.match("^yes$", Options["--membrane"], re.I) or re.match("^yes$", Options["--waterBox"], re.I):
            DeleteHeterogens = "WaterOnly"
    else:
        MiscUtil.PrintError(
            'The value specified, %s, for option "--deleteHeterogens" is not valid. Supported values: All AllExceptWater WaterOnly None'
            % DeleteHeterogens
        )
    OptionsInfo["DeleteHeterogens"] = DeleteHeterogens

    OptionsInfo["IonPositive"] = Options["--ionPositive"]
    OptionsInfo["IonNegative"] = Options["--ionNegative"]
    OptionsInfo["IonicStrength"] = float(Options["--ionicStrength"])

    OptionsInfo["ListDetails"] = True if re.match("^yes$", Options["--listDetails"], re.I) else False

    OptionsInfo["Membrane"] = True if re.match("^yes$", Options["--membrane"], re.I) else False
    ProcessMembraneParamsOption()

    OptionsInfo["ReplaceNonStandardResidues"] = (
        True if re.match("^yes$", Options["--replaceNonStandardResidues"], re.I) else False
    )

    OptionsInfo["WaterBox"] = True if re.match("^yes$", Options["--waterBox"], re.I) else False
    OptionsInfo["WaterBoxParams"] = OpenMMUtil.ProcessOptionOpenMMWaterBoxParameters(
        "--waterBoxParams", Options["--waterBoxParams"]
    )

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

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "pdb cif")
    MiscUtil.ValidateOptionsOutputFileOverwrite(
        "-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"]
    )
    MiscUtil.ValidateOptionsDistinctFileNames(
        "-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"]
    )

    MiscUtil.ValidateOptionTextValue("--addHeavyAtoms", Options["--addHeavyAtoms"], "yes no")

    MiscUtil.ValidateOptionTextValue("--addHydrogens", Options["--addHydrogens"], "yes no")
    MiscUtil.ValidateOptionFloatValue("--addHydrogensAtpH", Options["--addHydrogensAtpH"], {">=": 0})

    MiscUtil.ValidateOptionTextValue("--addResidues", Options["--addResidues"], "yes no")

    MiscUtil.ValidateOptionTextValue(
        "--deleteHeterogens", Options["--deleteHeterogens"], "All AllExceptWater WaterOnly None auto"
    )
    DeleteHeterogens = Options["--deleteHeterogens"]
    if re.match("^(AllExceptWater|None)$", DeleteHeterogens, re.I):
        if re.match("^yes$", Options["--membrane"], re.I):
            MiscUtil.PrintError(
                'The value specified, %s, for option "--deleteHeterogens" is not valid during "yes" value of option "--membrane". '
                % DeleteHeterogens
            )
        elif re.match("^yes$", Options["--waterBox"], re.I):
            MiscUtil.PrintError(
                'The value specified, %s, for option "--deleteHeterogens" is not valid during "yes" value of option "--waterBox". '
                % DeleteHeterogens
            )

    ValidValues = "Li+ Na+ K+ Rb+ Cs+"
    EscapedValidValuesPattern = r"Li\+|Na\+|K\+|Rb\+|Cs\+"
    Value = Options["--ionPositive"]
    if not re.match("^(%s)$" % EscapedValidValuesPattern, Value):
        MiscUtil.PrintError(
            'The value specified, %s, for option "-ionPositive" is not valid: Supported value(s): %s'
            % (Value, ValidValues)
        )

    ValidValues = "F- Cl- Br- I-"
    ValidValuesPattern = "F-|Cl-|Br-|I-"
    Value = Options["--ionNegative"]
    if not re.match("^(%s)$" % ValidValuesPattern, Value):
        MiscUtil.PrintError(
            'The value specified, %s, for option "-ionNegative" is not valid: Supported value(s): %s'
            % (Value, ValidValues)
        )

    MiscUtil.ValidateOptionFloatValue("--ionicStrength", Options["--ionicStrength"], {">=": 0})

    MiscUtil.ValidateOptionTextValue("--listDetails", Options["--listDetails"], "yes no")
    MiscUtil.ValidateOptionTextValue("--replaceNonStandardResidues", Options["--replaceNonStandardResidues"], "yes no")

    MiscUtil.ValidateOptionTextValue("--membrane", Options["--membrane"], "yes no")
    MiscUtil.ValidateOptionTextValue("--waterBox", Options["--waterBox"], "yes no")

    if re.match("^yes$", Options["--membrane"], re.I) and re.match("^yes$", Options["--waterBox"], re.I):
        MiscUtil.PrintError(
            'You\'ve specified, "Yes" for both "--membrane" and "--waterBox" options. It\'s not allowed. You must choose between adding a water box or a membrane. The water is automatically added during the construction of the membrane.'
        )


# Setup a usage string for docopt...
_docoptUsage_ = """
OpenMMPrepareMacromolecule.py - Prepare a macromolecule for simulation

Usage:
    OpenMMPrepareMacromolecule.py [--addHeavyAtoms <yes or no>] [--addHydrogens <yes or no>] [--addHydrogensAtpH <number>]
                                  [--addResidues <yes or no>] [--deleteHeterogens <All, AllExceptWater, WaterOnly, None>] [--ionPositive <text>]
                                  [--ionNegative <text>] [--ionicStrength <number>] [--listDetails <yes or no> ] [--membrane <yes or no>]
                                  [--membraneParams <Name,Value,..>] [--overwrite] [--replaceNonStandardResidues <yes or no>] [--waterBox <yes or no>]
                                  [--waterBoxParams <Name,Value,..>] [-w <dir>] -i <infile> -o <outfile>
    OpenMMPrepareMacromolecule.py -h | --help | -e | --examples

Description:
    Prepare a macromolecute in an input file for molecular simulation and
    and write it out to an output file. The macromolecule is prepared by
    automatically performing the following actions:

        . Identify and replace non-standard residues
        . Add missing residues
        . Add missing heavy atoms
        . Add missing hydrogens

    You may optionally remove heterogens, add a water box, and add a lipid
    membrane.

    The supported input file format are:  PDB (.pdb) and CIF (.cif)

    The supported output file formats are:  PDB (.pdb) and CIF (.cif)

Options:
    --addHeavyAtoms <yes or no>  [default: yes]
        Add missing non-hydrogen atoms based on the templates. The terminal atoms
        are also added.
    --addHydrogens <yes or no>  [default: yes]
        Add missing hydrogens at pH value specified using '-addHydrogensAtpH'
        option. The missing hydrogens are added based on the templates and pKa
        calculations are performed.
    --addHydrogensAtpH <number>  [default: 7.0]
        pH value to use for adding missing hydrogens.
    --addResidues <yes or no>  [default: yes]
        Add missing residues unidentified based on the PDB records.
    --deleteHeterogens <All, AllExceptWater, WaterOnly, None>  [default: auto]
        Delete heterogens corresponding to  non-standard names of amino acids, dna,
        and rna along with any ligand names. 'N' and 'UNK' also consider standard
        residues. Default value: WaterOnly during addition of WaterBox or Membrane;
        Otherwise, None.
        
        The 'AllExceptWater' or 'None' values are not allowed during the addition of
        a water box or membrane. The waters must be deleted as they are explicitly
        added during the construction of a water box and membrane.
    -e, --examples
        Print examples.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name.
    --ionPositive <text>  [default: Na+]
        Type of positive ion to add during the addition of a water box or membrane.
        Possible values: Li+, Na+, K+, Rb+, or Cs+.
    --ionNegative <text>  [default: Cl-]
        Type of negative ion to add during the addition of a water box or membrane.
        Possible values: Cl-, Br-, F-, or I-.
    --ionicStrength <number>  [default: 0.0]
        Total concentration of both positive and negative ions to add excluding
        the ions added to neutralize the system during the addition of a water box
        or a membrane.
    -l, --listDetails <yes or no>  [default: no]
        List details about missing and non-standard residues along with residues
        containing missing atoms.
    --membrane <yes or no>  [default: no]
        Add lipid membrane along with a water box. The script replies on OpenMM
        method modeller.addMebrane() to perform the task. The membrane is added
        in the XY plane. The existing macromolecule must be oriented and positioned
        correctly.
        
        A word to the wise: You may want to start with a model from the Orientations
        of Proteins in Membranes (OPM) database at http://opm.phar.umich.edu.
        
        The size of the membrane and water box are determined by the value of
        'padding' parameter specified using '--membraneParams' option. All atoms
        in macromolecule are guaranteed to be at least this far from any edge of
        the periodic box.
    --membraneParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for adding
        a lipid membrane and water.
        
        The supported parameter names along with their default values are
        are shown below:
            
            lipidType, POPC  [ Possible values: POPC, POPE, DLPC, DLPE,  DMPC, 
                DOPC or DPPC ]
            membraneCenterZ, 0.0
            padding, 1.0
            
        A brief description of parameters is provided below:
            
            lipidType: Type of lipid to use for constructing the membrane.
            membraneCenterZ: Position along the Z axis of the center of
                the membrane in nanomertes.
            padding: Minimum padding distance to use in nanomertes. It's used
                to determine the size of the membrane and water box. All atoms
                in the macromolecule are guaranteed to be at least this far from
                any edge of the periodic box.
            
    -o, --outfile <outfile>
        Output file name.
    --overwrite
        Overwrite existing files.
    --replaceNonStandardResidues <yes or no>  [default: yes]
        Replace non-standard residue names by standard residue names based on the
        list of non-standard residues available in pdbfixer.
    --waterBox <yes or no>  [default: no]
        Add water box.
    --waterBoxParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for adding
        a water box.
        
        The supported parameter names along with their default values are
        are shown below:
            
            mode, Padding  [ Possible values: Size or Padding ]
            size, None  [ Possible value: xsize ysize zsize ]
            padding, 1.0
            shape, cube  [ Possible values: cube, dodecahedron, or octahedron ]
            
        A brief description of parameters is provided below:
            
            mode: Specify the size of the waterbox explicitly or calculate it
                automatically for a macromolecule along with adding padding
                around ther macromolecule.
            size: A space delimited triplet of values corresponding to water
                size in nanometers. It must be specified during 'Size' value of
                'mode' parameter.
            padding: Padding around a macromolecule in nanometers for filling
                box with water. It must be specified during 'Padding' value of
                'mode' parameter.
            
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To prepare a macromolecule in a PDB file by replacing non-standard residues,
    adding missing residues, adding missing heavy atoms and missing hydrogens,
    and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py -i Sample11.pdb -o Sample11Out.pdb

    To run the first example for listing additional details about missing atoms and
    residues, and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py --listDetails yes -i Sample11.pdb
          -o Sample11Out.pdb

    To run the first example for deleting all heterogens including water along
    with performing all default actions, and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py --deleteHeterogens All -i Sample11.pdb
          -o Sample11Out.pdb

    To run the first example for deleting water only heterogens along with performing
    all default actions, and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py --deleteHeterogens WaterOnly
          -i Sample11.pdb -o Sample11Out.pdb --ov

    To run the first example for adding a water box by automatically calculating its
    size, along with performing all default actions, and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py --waterBox yes -i Sample11.pdb
          -o Sample11Out.pdb

    To run the previous example by explcitly specifying various water box parameters,
    and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py --waterBox yes
          --waterBoxParams "mode,Padding, padding, 1.0, shape, cube"
          -i Sample11.pdb -o Sample11Out.pdb

    To run the first example for adding a water box of a specific size along with
    performing all default actions, and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py --waterBox yes
          --waterBoxParams "mode,Size, size, 7.635 7.077 7.447"
          -i Sample11.pdb -o Sample11Out.pdb

    To add a lipid membrane around a correctly oriented and positioned macromolecule,
    along with performing all default actions,  and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py --membrane yes
          -i Sample12.pdb -o Sample12Out.pdb

    To add a lipid membrane around a correctly oriented and positioned macromolecule,
    deleting all heterogens along with performing all default actions,  and generate a
    PDB file, type:

        % OpenMMPrepareMacromolecule.py --membrane yes --deleteHeterogens All 
          -i Sample12.pdb -o Sample12Out.pdb

    To run the previous example by explcitly specifying various membrane parameters,
    and generate a PDB file, type:

        % OpenMMPrepareMacromolecule.py --membrane yes
          --membraneParams "lipidType, POPC, membraneCenterZ, 0.0, padding, 1.0"
          -i Sample12.pdb -o Sample12Out.pdb

Author:
    Manish Sud(msud@san.rr.com)

See also:
    InfoPDBFiles.pl, ExtractFromPDBFiles.pl, PyMOLExtractSelection.py,
    PyMOLInfoMacromolecules.py, PyMOLSplitChainsAndLigands.py

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
