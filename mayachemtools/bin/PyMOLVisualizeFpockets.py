#!/usr/bin/env python
#
# File: PyMOLVisualizeFpockets.py
# Author: Manish Sud <msud@san.rr.com>
#
# Author: Manish Sud
#
# Collaborators: Joann Prescott-Roy and Pat Walters
#
# Copyright (C) 2026 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using PyMOL, a
# molecular visualization system on an open source foundation originally
# developed by Warren DeLano.
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
#

from __future__ import print_function

import os
import sys
import time
import re
import glob
import shutil

# PyMOL imports...
try:
    import pymol

    # Finish launching PyMOL in  a command line mode for batch processing (-c)
    # along with the following options:  disable loading of pymolrc and plugins (-k);
    # suppress start up messages (-q)
    pymol.finish_launching(["pymol", "-ckq"])
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import PyMOL module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your PyMOL environment and try again.\n\n")
    sys.exit(1)

# MayaChemTools imports...
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), "..", "lib", "Python"))
try:
    from docopt import docopt
    import MiscUtil
    import PyMOLUtil
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import MayaChemTools module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your MayaChemTools environment and try again.\n\n")
    sys.exit(1)

ScriptName = os.path.basename(sys.argv[0])
Options = {}
OptionsInfo = {}


def main():
    """Start execution of the script"""

    MiscUtil.PrintInfo(
        "\n%s (PyMOL v%s; MayaChemTools v%s; %s): Starting...\n"
        % (ScriptName, pymol.cmd.get_version()[0], MiscUtil.GetMayaChemToolsVersion(), time.asctime())
    )

    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()

    # Retrieve command line arguments and options...
    RetrieveOptions()

    # Process and validate command line arguments and options...
    ProcessOptions()

    # Perform actions required by the script...
    GenerateFpocketsVisualization()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def GenerateFpocketsVisualization():
    """Generate fpockets visualization."""

    Outfile = OptionsInfo["PMLOutfilePath"]
    OutFH = open(Outfile, "w")
    if OutFH is None:
        MiscUtil.PrintError("Failed to open output fie %s " % Outfile)

    MiscUtil.PrintInfo("\nGenerating file %s..." % Outfile)

    # Setup header...
    WritePMLHeader(OutFH, ScriptName)
    WritePyMOLParameters(OutFH)

    if OptionsInfo["Align"]:
        WriteAlignReference(OutFH)

    # Setup view for each input file...
    FirstComplex = True
    FirstComplexFirstChainName = None
    for FileIndex in range(0, len(OptionsInfo["InfilesInfo"]["InfilesNames"])):
        # Setup PyMOL object names...
        PyMOLObjectNames = SetupPyMOLObjectNames(FileIndex)

        # Setup complex view...
        WriteComplexView(OutFH, FileIndex, PyMOLObjectNames, FirstComplex)

        # Setup chain and pocket views...
        SpecifiedChainsAndPocketsInfo = OptionsInfo["FpocketInfilesInfo"]["SpecifiedChainsAndPocketsInfo"][FileIndex]
        FirstChain = True
        for ChainID in SpecifiedChainsAndPocketsInfo["ChainIDs"]:
            if FirstComplex and FirstChain:
                FirstComplexFirstChainName = PyMOLObjectNames["Chains"][ChainID]["ChainAlone"]

            WriteChainView(OutFH, FileIndex, PyMOLObjectNames, ChainID)

            # Setup fpocket views...
            FirstPocket = True
            PocketNum = 0
            for PocketID in SpecifiedChainsAndPocketsInfo["PocketIDs"][ChainID]:
                PocketNum += 1
                WriteChainPocketView(OutFH, FileIndex, PyMOLObjectNames, ChainID, PocketID, PocketNum)

                # Set up pocket level group...
                Enable, Action = [False, "close"]
                if FirstPocket:
                    FirstPocket = False
                    Enable, Action = [True, "open"]
                GenerateAndWritePMLForGroup(
                    OutFH,
                    PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroup"],
                    PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroupMembers"],
                    Enable,
                    Action,
                )

            # Setup Chain level group...
            Enable, Action = [False, "close"]
            if FirstChain:
                FirstChain = False
                Enable, Action = [True, "open"]
            GenerateAndWritePMLForGroup(
                OutFH,
                PyMOLObjectNames["Chains"][ChainID]["ChainGroup"],
                PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"],
                Enable,
                Action,
            )

        # Set up complex level group...
        Enable, Action = [False, "close"]
        if FirstComplex:
            FirstComplex = False
            Enable, Action = [True, "open"]
        GenerateAndWritePMLForGroup(
            OutFH, PyMOLObjectNames["PDBGroup"], PyMOLObjectNames["PDBGroupMembers"], Enable, Action
        )

    if OptionsInfo["Align"]:
        DeleteAlignReference(OutFH)

    if FirstComplexFirstChainName is not None:
        OutFH.write("""\ncmd.orient("%s", animate = -1)\n""" % FirstComplexFirstChainName)
    else:
        OutFH.write("""\ncmd.orient("visible", animate = -1)\n""")

    OutFH.close()

    CopyPDBFilesForPML()


def WritePMLHeader(OutFH, ScriptName):
    """Write out PML header."""

    HeaderInfo = PyMOLUtil.SetupPMLHeaderInfo(ScriptName)
    OutFH.write("%s\n" % HeaderInfo)


def WritePyMOLParameters(OutFH):
    """Write out PyMOL global parameters."""

    PMLCmds = []
    PMLCmds.append("""cmd.set("transparency", %.2f, "", 0)""" % (OptionsInfo["SurfaceTransparency"]))
    PMLCmds.append("""cmd.set("label_font_id", %s)""" % (OptionsInfo["LabelFontID"]))
    PML = "\n".join(PMLCmds)

    OutFH.write("""\n""\n"Setting up PyMOL gobal parameters..."\n""\n""")
    OutFH.write("%s\n" % PML)


def WriteAlignReference(OutFH):
    """Setup object for alignment reference"""

    RefFileInfo = OptionsInfo["RefFileInfo"]
    RefFile = RefFileInfo["RefFileName"]
    RefName = RefFileInfo["PyMOLObjectName"]

    PMLCmds = []
    PMLCmds.append("""cmd.load("%s", "%s")""" % (RefFile, RefName))
    PMLCmds.append("""cmd.hide("everything", "%s")""" % (RefName))
    PMLCmds.append("""cmd.disable("%s")""" % (RefName))
    PML = "\n".join(PMLCmds)

    OutFH.write("""\n""\n"Loading %s and setting up view for align reference..."\n""\n""" % RefFile)
    OutFH.write("%s\n" % PML)


def WriteAlignComplex(OutFH, FileIndex, FpocketComplexMode, PyMOLObjectNames):
    """Setup alignment of complex to reference"""

    RefFileInfo = OptionsInfo["RefFileInfo"]
    RefName = RefFileInfo["PyMOLObjectName"]

    if FpocketComplexMode:
        ComplexName = PyMOLObjectNames["FpocketComplex"]
    else:
        ComplexName = PyMOLObjectNames["InitialComplex"]

    if re.match("^FirstChain$", OptionsInfo["AlignMode"], re.I):
        RefFirstChainID = RefFileInfo["ChainsAndLigandsInfo"]["ChainIDs"][0]
        RefAlignSelection = "%s and chain %s" % (RefName, RefFirstChainID)

        ComplexFirstChainID = RetrieveFirstChainID(FileIndex, FpocketComplexMode)
        ComplexAlignSelection = "%s and chain %s" % (ComplexName, ComplexFirstChainID)
    else:
        RefAlignSelection = RefName
        ComplexAlignSelection = ComplexName

    PML = PyMOLUtil.SetupPMLForAlignment(OptionsInfo["AlignMethod"], RefAlignSelection, ComplexAlignSelection)
    OutFH.write("""\n""\n"Aligning %s against reference %s ..."\n""\n""" % (ComplexAlignSelection, RefAlignSelection))
    OutFH.write("%s\n" % PML)


def DeleteAlignReference(OutFH):
    """Delete alignment reference object."""

    RefName = OptionsInfo["RefFileInfo"]["PyMOLObjectName"]
    OutFH.write("""\n""\n"Deleting alignment reference object %s..."\n""\n""" % RefName)
    OutFH.write("""cmd.delete("%s")\n""" % RefName)


def WriteComplexView(OutFH, FileIndex, PyMOLObjectNames, FirstComplex):
    """Write out PML for viewing polymer complex."""

    # Setup initial complex...
    Infile = OptionsInfo["InfilesInfo"]["InfilesNames"][FileIndex]
    PML = PyMOLUtil.SetupPMLForPolymerComplexView(PyMOLObjectNames["InitialComplex"], Infile, True)
    OutFH.write("""\n""\n"Loading %s and setting up view for complex..."\n""\n""" % Infile)
    OutFH.write("%s\n" % PML)

    if OptionsInfo["Align"]:
        # No need to align initial complex on to itself...
        FpocketComplexMode = False
        if not (re.match("^FirstInputFile$", OptionsInfo["AlignRefFile"], re.I) and FirstComplex):
            WriteAlignComplex(OutFH, FileIndex, FpocketComplexMode, PyMOLObjectNames)

    # Setup fpocket complex...
    Infile = OptionsInfo["FpocketInfilesInfo"]["InfilesNames"][FileIndex]
    PML = PyMOLUtil.SetupPMLForPolymerComplexView(PyMOLObjectNames["FpocketComplex"], Infile, True)

    # Modify default complex view for fpocket complex...
    PMLModify = SetupPMLModifyDefaultPolymerComplexView(PyMOLObjectNames["FpocketComplex"])

    OutFH.write("""\n""\n"Loading %s and setting up view for complex..."\n""\n""" % Infile)
    OutFH.write("%s\n%s\n" % (PML, PMLModify))

    if OptionsInfo["Align"]:
        # No need to align initial complex on to itself...
        FpocketComplexMode = True
        if not (re.match("^FirstInputFile$", OptionsInfo["AlignRefFile"], re.I) and FirstComplex):
            WriteAlignComplex(OutFH, FileIndex, FpocketComplexMode, PyMOLObjectNames)

    # Setup complex group...
    GenerateAndWritePMLForGroup(
        OutFH, PyMOLObjectNames["ComplexGroup"], PyMOLObjectNames["ComplexGroupMembers"], False, "close"
    )


def WriteChainView(OutFH, FileIndex, PyMOLObjectNames, ChainID):
    """Write out PML for viewing chain."""

    OutFH.write("""\n""\n"Setting up views for chain %s..."\n""\n""" % ChainID)

    # Setup chain complex group view...
    WriteChainComplexViews(OutFH, FileIndex, PyMOLObjectNames, ChainID)

    # Setup chain view...
    WriteChainAloneViews(OutFH, FileIndex, PyMOLObjectNames, ChainID)


def WriteChainComplexViews(OutFH, FileIndex, PyMOLObjectNames, ChainID):
    """Write chain complex views."""

    # Setup chain complex...
    ChainComplexName = PyMOLObjectNames["Chains"][ChainID]["ChainComplex"]
    PML = SetupPMLForFpocketChainComplexView(
        FileIndex, ChainComplexName, PyMOLObjectNames["FpocketComplex"], ChainID, True
    )

    # Modify default complex view for fpocket chain complex...
    PMLModify = SetupPMLModifyDefaultPolymerComplexView(ChainComplexName)
    OutFH.write("%s\n%s\n" % (PML, PMLModify))

    # Setup chain complex group...
    GenerateAndWritePMLForGroup(
        OutFH,
        PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroup"],
        PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroupMembers"],
        False,
        "close",
    )


def WriteChainAloneViews(OutFH, FileIndex, PyMOLObjectNames, ChainID):
    """Write individual chain views."""

    ChainComplexName = PyMOLObjectNames["Chains"][ChainID]["ChainComplex"]

    # Setup chain view...
    ChainName = PyMOLObjectNames["Chains"][ChainID]["ChainAlone"]
    PML = PyMOLUtil.SetupPMLForPolymerChainView(ChainName, ChainComplexName, True)
    OutFH.write("\n%s\n" % PML)

    if GetChainAloneContainsSurfacesStatus(FileIndex, ChainID):
        # Setup a generic color surface...
        PML = PyMOLUtil.SetupPMLForSurfaceView(
            PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurface"],
            ChainName,
            Enable=False,
            Color=OptionsInfo["SurfaceColor"],
        )
        OutFH.write("\n%s\n" % PML)

        if GetChainAloneSurfaceChainStatus(FileIndex, ChainID):
            # Setup surface colored by hydrophobicity...
            PML = PyMOLUtil.SetupPMLForHydrophobicSurfaceView(
                PyMOLObjectNames["Chains"][ChainID]["ChainAloneHydrophobicSurface"],
                ChainName,
                ColorPalette=OptionsInfo["SurfaceColorPalette"],
                Enable=False,
            )
            OutFH.write("\n%s\n" % PML)

            # Setup surface colored by hyrdophobicity and charge...
            PML = PyMOLUtil.SetupPMLForHydrophobicAndChargeSurfaceView(
                PyMOLObjectNames["Chains"][ChainID]["ChainAloneHydrophobicChargeSurface"],
                ChainName,
                OptionsInfo["AtomTypesColorNames"]["HydrophobicAtomsColor"],
                OptionsInfo["AtomTypesColorNames"]["NegativelyChargedAtomsColor"],
                OptionsInfo["AtomTypesColorNames"]["PositivelyChargedAtomsColor"],
                OptionsInfo["AtomTypesColorNames"]["OtherAtomsColor"],
                Enable=False,
                DisplayAs=None,
            )
            OutFH.write("\n%s\n" % PML)

        # Setup surface group...
        GenerateAndWritePMLForGroup(
            OutFH,
            PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurfaceGroup"],
            PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurfaceGroupMembers"],
            True,
            "open",
        )

    # Setup chain group...
    GenerateAndWritePMLForGroup(
        OutFH,
        PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroup"],
        PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"],
        True,
        "close",
    )


def WriteChainPocketView(OutFH, FileIndex, PyMOLObjectNames, ChainID, PocketID, PocketNum):
    """Write out PML for viewing pocket in a chain."""

    OutFH.write("""\n""\n"Setting up views for pocket %s in chain %s..."\n""\n""" % (PocketID, ChainID))

    FpocketComplexName = PyMOLObjectNames["FpocketComplex"]

    # Setup pocket...
    PML = SetupPMLForFPocketView(
        PyMOLObjectNames["Pockets"][ChainID][PocketID]["Pocket"], FpocketComplexName, ChainID, PocketID, PocketNum, True
    )
    OutFH.write("%s\n" % PML)

    # Setup pocket residues...
    ChainsAndPocketsInfo = OptionsInfo["FpocketInfilesInfo"]["ChainsAndPocketsInfo"][FileIndex]
    PocketResNums = ChainsAndPocketsInfo["PocketResNums"][ChainID][PocketID]
    PocketResiduesName = PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketResidues"]
    PML = SetupPMLForFPocketResiduesView(
        PocketResiduesName, FpocketComplexName, ChainID, PocketID, PocketNum, PocketResNums, True
    )
    OutFH.write("%s\n" % PML)

    # Setup pocket surfaces and group...
    if GetPocketContainsSurfaceStatus(FileIndex, ChainID, PocketID):
        # Setup a generic color surface...
        PML = PyMOLUtil.SetupPMLForSurfaceView(
            PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurface"],
            PocketResiduesName,
            Enable=False,
            Color=OptionsInfo["SurfaceColor"],
        )
        OutFH.write("\n%s\n" % PML)

        if GetPocketSurfaceChainStatus(FileIndex, ChainID, PocketID):
            # Setup surface colored by hydrophobicity...
            PML = PyMOLUtil.SetupPMLForHydrophobicSurfaceView(
                PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketHydrophobicitySurface"],
                PocketResiduesName,
                ColorPalette=OptionsInfo["SurfaceColorPalette"],
                Enable=False,
            )
            OutFH.write("\n%s\n" % PML)

            # Setup surface colored by hyrdophobicity and charge...
            PML = PyMOLUtil.SetupPMLForHydrophobicAndChargeSurfaceView(
                PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketHydrophobicityChargeSurface"],
                PocketResiduesName,
                OptionsInfo["AtomTypesColorNames"]["HydrophobicAtomsColor"],
                OptionsInfo["AtomTypesColorNames"]["NegativelyChargedAtomsColor"],
                OptionsInfo["AtomTypesColorNames"]["PositivelyChargedAtomsColor"],
                OptionsInfo["AtomTypesColorNames"]["OtherAtomsColor"],
                Enable=False,
                DisplayAs=None,
            )
            OutFH.write("\n%s\n" % PML)

        # Setup surface group...
        GenerateAndWritePMLForGroup(
            OutFH,
            PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurfaceGroup"],
            PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurfaceGroupMembers"],
            True,
            "open",
        )

    # Setup pocket group...
    GenerateAndWritePMLForGroup(
        OutFH,
        PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroup"],
        PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroupMembers"],
        True,
        "open",
    )


def SetupPMLForFpocketChainComplexView(FileIndex, Name, Selection, ChainName, Enable=True):
    """Setup PML commands for creating a polymer chain complex view
    including fockets for the chain."""

    PMLCmds = []

    # Include fpockets as spheres for the chain complex view...
    ChainsAndPocketsInfo = OptionsInfo["FpocketInfilesInfo"]["ChainsAndPocketsInfo"][FileIndex]
    PocketIDs = ChainsAndPocketsInfo["PocketIDs"][ChainName]

    PMLCmds.append(
        """cmd.create("%s", "((%s and chain %s) or (%s and (resn STP and resi %s)))")"""
        % (Name, Selection, ChainName, Selection, "+".join(PocketIDs))
    )
    PMLCmds.append("""cmd.hide("everything", "%s")""" % (Name))
    PMLCmds.append("""cmd.show("cartoon", "%s")""" % (Name))
    PMLCmds.append("""util.cba(33, "%s", _self = cmd)""" % (Name))
    PMLCmds.append("""cmd.show("sticks", "(organic and (%s))")""" % (Name))

    PMLCmds.append("""cmd.show("nonbonded", "(solvent and (%s))")""" % (Name))
    PMLCmds.append("""cmd.show("nonbonded", "(inorganic and (%s))")""" % (Name))

    PMLCmds.append("""cmd.show("lines", "%s")""" % (Name))

    PMLCmds.append("""cmd.set_bond("valence", "1", "%s", quiet = 1)""" % (Name))
    PMLCmds.append(PyMOLUtil.SetupPMLForEnableDisable(Name, Enable))

    PML = "\n".join(PMLCmds)

    return PML


def SetupPMLModifyDefaultPolymerComplexView(Name):
    """Setup PML to modify default polymer complex view for fpocket."""

    PMLCmds = []
    PMLCmds.append("""cmd.hide("lines", "%s")""" % (Name))
    PMLCmds.append("""cmd.show("lines", "(polymer and (%s))")""" % (Name))
    PMLCmds.append("""cmd.show("lines", "(solvent and (%s))")""" % (Name))
    PMLCmds.append("""cmd.show("spheres", "(inorganic and (%s))")""" % (Name))
    PMLCmds.append("""cmd.set("sphere_scale", "%s", "(inorganic and (%s))")""" % (OptionsInfo["SphereScale"], Name))
    PMLCmds.append(
        """cmd.set("sphere_transparency", "%s", "(inorganic and (%s))")""" % (OptionsInfo["SphereTransparency"], Name)
    )

    PML = "\n".join(PMLCmds)

    return PML


def SetupPMLForFPocketView(FpocketName, FpocketComplexName, ChainID, PocketID, PocketNum, Enable=True):
    """Setup PML  to visualize fpocket spheres using fpocket complex."""

    PMLCmds = []
    PMLCmds.append(
        """cmd.create("%s", "(%s and (resn STP and resi %s))")""" % (FpocketName, FpocketComplexName, PocketID)
    )
    if PocketNum == 1:
        # Skip color index of 1: It is set to black. Use pocket one color name...
        PMLCmds.append("""cmd.color(%s, "%s")""" % (OptionsInfo["PocketNumOneColor"], FpocketName))
    else:
        PMLCmds.append("""cmd.color(%s, "%s")""" % (PocketNum, FpocketName))
    PMLCmds.append("""cmd.show("spheres", "%s")""" % (FpocketName))
    PMLCmds.append("""cmd.set("sphere_scale", "%s", "%s")""" % (OptionsInfo["SphereScale"], FpocketName))
    PMLCmds.append("""cmd.set("sphere_transparency", "%s", "%s")""" % (OptionsInfo["SphereTransparency"], FpocketName))

    PMLCmds.append(PyMOLUtil.SetupPMLForEnableDisable(FpocketName, Enable))

    PML = "\n".join(PMLCmds)
    return PML


def SetupPMLForFPocketResiduesView(Name, FpocketComplexName, ChainID, PocketID, PocketNum, PocketResNums, Enable=True):
    """Setup PML to visualize fpocket residues."""

    PMLCmds = []

    PMLCmds.append(
        """cmd.create("%s", "(%s and (chain %s) and (resi %s))")"""
        % (Name, FpocketComplexName, ChainID, "+".join(PocketResNums))
    )
    PMLCmds.append("""cmd.hide("everything", "%s")""" % (Name))

    # Setup pocket residue labels...
    if OptionsInfo["PocketLabel"]:
        if OptionsInfo["ThreeLetterPocketLabelType"]:
            LabelFormat = """"%s-%s"%(resn,resi)"""
            PMLCmds.append("""cmd.label("(name CA+C1*+C1' and (byres(%s)))", \'\'\'%s\'\'\')""" % (Name, LabelFormat))
        else:
            PMLCmds.append("""cmd.label("byca(%s)", "oneletter+resi")""" % (Name))

    PocketColor = OptionsInfo["PocketNumOneColor"] if PocketNum == 1 else PocketNum

    if OptionsInfo["PocketColorByPocketNum"]:
        # Setup color of pocket residues...
        PMLCmds.append(PyMOLUtil.SetupPMLForDeepColoring(Name, PocketColor))

        # Setup color of pocket residue labels...
        PMLCmds.append("""cmd.set("label_color", %s, "%s")""" % (PocketColor, Name))
    else:
        PMLCmds.append("""util.cbag("%s", _self = cmd)""" % (Name))

    PMLCmds.append("""cmd.show("lines", "%s")""" % (Name))

    PMLCmds.append(PyMOLUtil.SetupPMLForEnableDisable(Name, Enable))

    PML = "\n".join(PMLCmds)
    return PML


def GenerateAndWritePMLForGroup(OutFH, GroupName, GroupMembers, Enable=False, Action="close"):
    """Generate and write PML for group."""

    PML = PyMOLUtil.SetupPMLForGroup(GroupName, GroupMembers, Enable, Action)
    OutFH.write("""\n""\n"Setting up group %s..."\n""\n""" % GroupName)
    OutFH.write("%s\n" % PML)


def WritePMLToCheckAndDeleteEmptyObjects(OutFH, ObjectName, ParentObjectName=None):
    """Write PML to check and delete empty PyMOL objects."""

    if ParentObjectName is None:
        PML = """CheckAndDeleteEmptyObjects("%s")""" % (ObjectName)
    else:
        PML = """CheckAndDeleteEmptyObjects("%s", "%s")""" % (ObjectName, ParentObjectName)

    OutFH.write("%s\n" % PML)


def SetupPyMOLObjectNames(FileIndex):
    """Setup hierarchy of PyMOL groups and objects for pocket centric views of
    chains and pockets present in input file.
    """

    PyMOLObjectNames = {}
    PyMOLObjectNames["Chains"] = {}
    PyMOLObjectNames["Pockets"] = {}

    # Setup groups and objects for complex...
    SetupPyMOLObjectNamesForComplex(FileIndex, PyMOLObjectNames)

    # Setup groups and objects for chains using fpockets info...
    SpecifiedChainsAndPocketsInfo = OptionsInfo["FpocketInfilesInfo"]["SpecifiedChainsAndPocketsInfo"][FileIndex]
    for ChainID in SpecifiedChainsAndPocketsInfo["ChainIDs"]:
        SetupPyMOLObjectNamesForChain(FileIndex, PyMOLObjectNames, ChainID)

        # Setup groups and objects for pocket...
        for PocketID in SpecifiedChainsAndPocketsInfo["PocketIDs"][ChainID]:
            SetupPyMOLObjectNamesForPocket(FileIndex, PyMOLObjectNames, ChainID, PocketID)

    return PyMOLObjectNames


def SetupPyMOLObjectNamesForComplex(FileIndex, PyMOLObjectNames):
    """Setup groups and objects for complex."""

    PDBFileRoot = OptionsInfo["InfilesInfo"]["InfilesRoots"][FileIndex]

    PDBGroupName = "%s" % PDBFileRoot
    PyMOLObjectNames["PDBGroup"] = PDBGroupName
    PyMOLObjectNames["PDBGroupMembers"] = []

    ComplexGroupName = "%s.Complex" % PyMOLObjectNames["PDBGroup"]
    PyMOLObjectNames["ComplexGroup"] = ComplexGroupName
    PyMOLObjectNames["ComplexGroupMembers"] = []

    PyMOLObjectNames["PDBGroupMembers"].append(ComplexGroupName)

    PyMOLObjectNames["InitialComplex"] = "%s.Initial_Complex" % ComplexGroupName
    PyMOLObjectNames["FpocketComplex"] = "%s.Fpocket_Complex" % ComplexGroupName

    PyMOLObjectNames["ComplexGroupMembers"].append(PyMOLObjectNames["InitialComplex"])
    PyMOLObjectNames["ComplexGroupMembers"].append(PyMOLObjectNames["FpocketComplex"])


def SetupPyMOLObjectNamesForChain(FileIndex, PyMOLObjectNames, ChainID):
    """Setup groups and objects for chain."""

    PDBGroupName = PyMOLObjectNames["PDBGroup"]

    PyMOLObjectNames["Chains"][ChainID] = {}
    PyMOLObjectNames["Pockets"][ChainID] = {}

    # Set up chain group and chain objects...
    ChainGroupName = "%s.Chain%s" % (PDBGroupName, ChainID)
    PyMOLObjectNames["Chains"][ChainID]["ChainGroup"] = ChainGroupName
    PyMOLObjectNames["PDBGroupMembers"].append(ChainGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"] = []

    # Setup chain complex group and objects...
    ChainComplexGroupName = "%s.Complex" % (ChainGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroup"] = ChainComplexGroupName
    PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"].append(ChainComplexGroupName)

    PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroupMembers"] = []

    Name = "%s.Complex" % (ChainComplexGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainComplex"] = Name
    PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroupMembers"].append(Name)

    # Setup up a group for individual chains...
    ChainAloneGroupName = "%s.Chain" % (ChainGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroup"] = ChainAloneGroupName
    PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"].append(ChainAloneGroupName)

    PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"] = []

    Name = "%s.Chain" % (ChainAloneGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainAlone"] = Name
    PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"].append(Name)

    if GetChainAloneContainsSurfacesStatus(FileIndex, ChainID):
        # Setup a surface group and add it to chain alone group...
        SurfaceGroupName = "%s.Surface" % (ChainAloneGroupName)
        PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurfaceGroup"] = SurfaceGroupName
        PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"].append(SurfaceGroupName)

        PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurfaceGroupMembers"] = []

        # Setup a generic color surface...
        Name = "%s.Surface" % (SurfaceGroupName)
        PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurface"] = Name
        PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurfaceGroupMembers"].append(Name)

        if GetChainAloneSurfaceChainStatus(FileIndex, ChainID):
            # Setup hydrophobicity surface...
            Name = "%s.Hydrophobicity" % (SurfaceGroupName)
            PyMOLObjectNames["Chains"][ChainID]["ChainAloneHydrophobicSurface"] = Name
            PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurfaceGroupMembers"].append(Name)

            # Setup hydrophobicity and charge surface...
            Name = "%s.Hydrophobicity_Charge" % (SurfaceGroupName)
            PyMOLObjectNames["Chains"][ChainID]["ChainAloneHydrophobicChargeSurface"] = Name
            PyMOLObjectNames["Chains"][ChainID]["ChainAloneSurfaceGroupMembers"].append(Name)


def SetupPyMOLObjectNamesForPocket(FileIndex, PyMOLObjectNames, ChainID, PocketID):
    """Stetup groups and objects for pocket."""

    PyMOLObjectNames["Pockets"][ChainID][PocketID] = {}

    ChainGroupName = PyMOLObjectNames["Chains"][ChainID]["ChainGroup"]

    # Setup a chain level pocket group...
    ChainPocketGroupName = SetupChainPocketGroupName(FileIndex, ChainGroupName, ChainID, PocketID)

    PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroup"] = ChainPocketGroupName
    PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"].append(ChainPocketGroupName)

    PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroupMembers"] = []

    # Setup fpocket...
    Name = "%s.Fpocket" % (ChainPocketGroupName)
    PyMOLObjectNames["Pockets"][ChainID][PocketID]["Pocket"] = Name
    PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroupMembers"].append(Name)

    # Setup fpocket residues...
    Name = "%s.Residues" % (ChainPocketGroupName)
    PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketResidues"] = Name
    PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroupMembers"].append(Name)

    if GetPocketContainsSurfaceStatus(FileIndex, ChainID, PocketID):
        # Setup a pocket surface group and add it to chain pocket group
        SurfaceGroupName = "%s.Surface" % (ChainPocketGroupName)
        PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurfaceGroup"] = SurfaceGroupName
        PyMOLObjectNames["Pockets"][ChainID][PocketID]["ChainPocketGroupMembers"].append(SurfaceGroupName)

        PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurfaceGroupMembers"] = []

        # Setup a generic color surface...
        Name = "%s.Surface" % (SurfaceGroupName)
        PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurface"] = Name
        PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurfaceGroupMembers"].append(Name)

        if GetPocketSurfaceChainStatus(FileIndex, ChainID, PocketID):
            # Surface colored by hydrophobicity...
            Name = "%s.Hydrophobicity" % (SurfaceGroupName)
            PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketHydrophobicitySurface"] = Name
            PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurfaceGroupMembers"].append(Name)

            # Surface colored by hydrophobicity and charge...
            Name = "%s.Hydrophobicity_Charge" % (SurfaceGroupName)
            PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketHydrophobicityChargeSurface"] = Name
            PyMOLObjectNames["Pockets"][ChainID][PocketID]["PocketSurfaceGroupMembers"].append(Name)


def SetupChainPocketGroupName(FileIndex, ChainGroupName, ChainID, PocketID):
    """Setup pocket group PyMOL object name."""

    ChainPocketGroupName = "%s.Fpocket%s" % (ChainGroupName, PocketID)

    if not OptionsInfo["FpocketPropertiesAppend"]:
        return ChainPocketGroupName

    PocketPropertiesName = SetupFpocketPropertiesForGroupName(FileIndex, ChainGroupName, ChainID, PocketID)

    ChainPocketGroupName = "%s_%s" % (ChainPocketGroupName, PocketPropertiesName)

    return ChainPocketGroupName


def SetupFpocketPropertiesForGroupName(FileIndex, ChainGroupName, ChainID, PocketID):
    """Setup fpocket properties for PyMOL group name."""

    ChainsAndPocketsInfo = OptionsInfo["FpocketInfilesInfo"]["ChainsAndPocketsInfo"][FileIndex]

    PocketScore = FormatFpocketPropertyForGroupName(ChainsAndPocketsInfo["PocketScore"][ChainID][PocketID])
    DrugScore = FormatFpocketPropertyForGroupName(ChainsAndPocketsInfo["DrugScore"][ChainID][PocketID])
    HydrophobicityScore = FormatFpocketPropertyForGroupName(
        ChainsAndPocketsInfo["HydrophobicityScore"][ChainID][PocketID]
    )
    PolarityScore = FormatFpocketPropertyForGroupName(ChainsAndPocketsInfo["PolarityScore"][ChainID][PocketID])
    PocketVolume = FormatFpocketPropertyForGroupName(ChainsAndPocketsInfo["PocketVolume"][ChainID][PocketID])

    PocketProperties = "S%s_D%s_V%s_H%s_P%s" % (
        PocketScore,
        DrugScore,
        PocketVolume,
        HydrophobicityScore,
        PolarityScore,
    )

    return PocketProperties


def FormatFpocketPropertyForGroupName(PocketProperty):
    """Format fpocket property for PyMOL group name."""

    if PocketProperty is None:
        PocketProperty = "NA"
    else:
        PocketProperty = "%.2f" % float(PocketProperty)
        PocketProperty = re.sub(r"\.", "p", PocketProperty)
        PocketProperty = re.sub("^-", "neg", PocketProperty)

    return PocketProperty


def GetChainAloneContainsSurfacesStatus(FileIndex, ChainID):
    """Get status of surfaces present in chain alone object."""

    # Always set up generic color surfaces...
    return True


def GetChainAloneSurfaceChainStatus(FileIndex, ChainID):
    """Get status of  surfaces for chain alone object."""

    return OptionsInfo["SurfaceChain"]


def GetPocketContainsSurfaceStatus(FileIndex, ChainID, PocketID):
    """Get status of surfaces present in  a pocket object."""

    # Always set up generic color surfaces...
    return True


def GetPocketSurfaceChainStatus(FileIndex, ChainID, PocketID):
    """Get status of  surfaces for pocket object."""

    return OptionsInfo["PocketSurface"]


def CopyPDBFilesForPML():
    """Copy appropriate PDB files for PyMOL to OutfilesDir"""

    OutfilesDir = OptionsInfo["OutfilesDir"]

    MiscUtil.PrintInfo("\nCopying appropriate PDB files to directory %s..." % OutfilesDir)

    # Copy input PDB files...
    for FileIndex in range(0, len(OptionsInfo["InfilesInfo"]["InfilesNames"])):
        Infile = OptionsInfo["InfilesInfo"]["InfilesNames"][FileIndex]
        NewInfilePath = os.path.join(OutfilesDir, Infile)

        shutil.copyfile(Infile, NewInfilePath)

    # Copy fpocket output PDB files...
    for FileIndex in range(0, len(OptionsInfo["FpocketInfilesInfo"]["InfilesNames"])):
        Infile = OptionsInfo["FpocketInfilesInfo"]["InfilesNames"][FileIndex]
        InfilePath = OptionsInfo["FpocketInfilesInfo"]["InfilesPaths"][FileIndex]
        NewInfilePath = os.path.join(OutfilesDir, Infile)

        shutil.copyfile(InfilePath, NewInfilePath)


def RetrieveInfilesInfo():
    """Retrieve information for input files."""

    InfilesInfo = {}

    InfilesInfo["InfilesNames"] = []
    InfilesInfo["InfilesRoots"] = []
    InfilesInfo["ChainsAndLigandsInfo"] = []

    for Infile in OptionsInfo["InfilesNames"]:
        FileDir, FileName, FileExt = MiscUtil.ParseFileName(Infile)
        InfileRoot = FileName

        ChainsAndLigandInfo = PyMOLUtil.GetChainsAndLigandsInfo(Infile, InfileRoot)

        InfilesInfo["InfilesNames"].append(Infile)
        InfilesInfo["InfilesRoots"].append(InfileRoot)
        InfilesInfo["ChainsAndLigandsInfo"].append(ChainsAndLigandInfo)

    OptionsInfo["InfilesInfo"] = InfilesInfo


def RetrieveRefFileInfo():
    """Retrieve information for ref file."""

    RefFileInfo = {}
    if not OptionsInfo["Align"]:
        OptionsInfo["RefFileInfo"] = RefFileInfo
        return

    RefFile = OptionsInfo["RefFileName"]

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(RefFile)
    RefFileRoot = FileName

    if re.match("^FirstInputFile$", OptionsInfo["AlignRefFile"], re.I):
        ChainsAndLigandInfo = OptionsInfo["InfilesInfo"]["ChainsAndLigandsInfo"][0]
    else:
        MiscUtil.PrintInfo("\nRetrieving chain and ligand information for alignment reference file %s..." % RefFile)
        ChainsAndLigandInfo = PyMOLUtil.GetChainsAndLigandsInfo(RefFile, RefFileRoot)

    RefFileInfo["RefFileName"] = RefFile
    RefFileInfo["RefFileRoot"] = RefFileRoot
    RefFileInfo["PyMOLObjectName"] = "AlignRef_%s" % RefFileRoot
    RefFileInfo["ChainsAndLigandsInfo"] = ChainsAndLigandInfo

    OptionsInfo["RefFileInfo"] = RefFileInfo


def RetrieveFpocketResultFilesInfo():
    """Check and retrieve information for Fpocket result files."""

    FpocketInfilesInfo = {}

    FpocketInfilesInfo["InfilesNames"] = []
    FpocketInfilesInfo["InfilesRoots"] = []
    FpocketInfilesInfo["InfilesPaths"] = []
    FpocketInfilesInfo["InfilesDirs"] = []

    FpocketInfilesInfo["InfilesPocketsDirs"] = []
    FpocketInfilesInfo["InfilesPocketsPDBFilesCount"] = []

    FpocketInfilesInfo["ChainsAndPocketsInfo"] = []

    for Infile in OptionsInfo["InfilesNames"]:
        MiscUtil.PrintInfo("\nRetrieving Fpocket result file information for input file %s..." % Infile)

        FileDir, FileName, FileExt = MiscUtil.ParseFileName(Infile)
        InfileRoot = FileName

        FpocketInfileDir = "%s_out" % InfileRoot
        FpocketInfileRoot = "%s_out" % InfileRoot
        FpocketInfileName = "%s.pdb" % FpocketInfileRoot
        FpocketInfilePath = os.path.join(FpocketInfileDir, FpocketInfileName)

        FpocketInfilePocketsDir = os.path.join(FpocketInfileDir, "pockets")

        if not os.path.isdir(FpocketInfileDir):
            MiscUtil.PrintError(
                "Fpocket result file directory, %s, is missing for input PDB file, %s." % (FpocketInfileDir, Infile)
            )

        if not os.path.isfile(FpocketInfilePath):
            MiscUtil.PrintError(
                "Fpocket result PDB file, %s, is missing for input PDB file, %s." % (FpocketInfilePath, Infile)
            )

        if not os.path.isdir(FpocketInfilePocketsDir):
            MiscUtil.PrintError(
                "Fpocket pockets result file directory, %s, is missing for input PDB file, %s."
                % (FpocketInfilePocketsDir, Infile)
            )

        PocketsPDBFiles = glob.glob(os.path.join(FpocketInfilePocketsDir, "pocket*_atm.pdb"))
        PocketsPDBFilesCount = len(PocketsPDBFiles)
        if PocketsPDBFilesCount == 0:
            MiscUtil.PrintError(
                "Fpocket pockets result files missing in directory, %s, for input PDB file, %s."
                % (FpocketInfilePocketsDir, Infile)
            )

        # Retrieve chains and pockets...
        ChainsAndPocketsInfo = GetFpocketChainsAndPocketsInfo(
            FpocketInfileRoot, FpocketInfilePath, FpocketInfileDir, FpocketInfilePocketsDir
        )

        FpocketInfilesInfo["InfilesNames"].append(FpocketInfileName)
        FpocketInfilesInfo["InfilesRoots"].append(FpocketInfileRoot)
        FpocketInfilesInfo["InfilesPaths"].append(FpocketInfilePath)
        FpocketInfilesInfo["InfilesDirs"].append(FpocketInfileDir)

        FpocketInfilesInfo["InfilesPocketsDirs"].append(FpocketInfilePocketsDir)
        FpocketInfilesInfo["InfilesPocketsPDBFilesCount"].append(PocketsPDBFilesCount)

        FpocketInfilesInfo["ChainsAndPocketsInfo"].append(ChainsAndPocketsInfo)

        MiscUtil.PrintInfo("PDB result file: %s..." % FpocketInfileName)
        MiscUtil.PrintInfo("Total number of pockets: %s..." % ChainsAndPocketsInfo["NumOfPockets"])

        MiscUtil.PrintInfo("Pocket Chain IDs: %s" % " ".join(ChainsAndPocketsInfo["ChainIDs"]))
        for ChainID in ChainsAndPocketsInfo["ChainIDs"]:
            MiscUtil.PrintInfo(
                "Pocket Chain ID: %s; NumOfPockets: %s; PocketIDs: %s"
                % (
                    ChainID,
                    len(ChainsAndPocketsInfo["PocketIDs"][ChainID]),
                    " ".join(ChainsAndPocketsInfo["PocketIDs"][ChainID]),
                )
            )

    OptionsInfo["FpocketInfilesInfo"] = FpocketInfilesInfo


def GetFpocketChainsAndPocketsInfo(FpocketInfileRoot, FpocketInfilePath, FpocketInfileDir, FpocketInfilePocketsDir):
    """Get chains and pockets information for Fpocket result files."""

    ChainsAndPocketsInfo = {}
    ChainsAndPocketsInfo["ChainIDs"] = []
    ChainsAndPocketsInfo["PocketIDs"] = {}
    ChainsAndPocketsInfo["PocketResNums"] = {}

    ChainsAndPocketsInfo["DrugScore"] = {}
    ChainsAndPocketsInfo["PocketScore"] = {}
    ChainsAndPocketsInfo["HydrophobicityScore"] = {}
    ChainsAndPocketsInfo["PolarityScore"] = {}
    ChainsAndPocketsInfo["PocketVolume"] = {}

    ChainsAndPocketsInfo["NumOfPockets"] = 0

    # Retrieve number of pockets and pocket IDs...
    PocketIDs = GetPocketIDs(FpocketInfilePath, FpocketInfileRoot)
    ChainsAndPocketsInfo["NumOfPockets"] = len(PocketIDs)

    # Retrieve residue numbers for fpockets across the chains...
    for PocketID in PocketIDs:
        PocketAtomFile = os.path.join(FpocketInfilePocketsDir, "pocket%s_atm.pdb" % PocketID)

        if not os.path.isfile(PocketAtomFile):
            MiscUtil.PrintError(
                "Fpocket result PDB file, %s, is missing for pocket ID, %s." % (PocketAtomFile, PocketID)
            )

        # Retrieve pocket properites for a pocket across chains...
        PocketScore, DrugScore, HydrophobicityScore, PolarityScore, PocketVolume = GetPocketProperties(PocketAtomFile)

        MolName = "pocket%s_atm" % PocketID
        pymol.cmd.load(PocketAtomFile, MolName)
        SelectionCmd = "(%s)" % MolName

        pymol.stored.FpocketInfo = []
        pymol.cmd.iterate(SelectionCmd, "pymol.stored.FpocketInfo.append([chain, resi, resn])")
        pymol.cmd.delete(MolName)

        for ChainID, ResNum, ResName in pymol.stored.FpocketInfo:
            if ChainID not in ChainsAndPocketsInfo["ChainIDs"]:
                ChainsAndPocketsInfo["PocketIDs"][ChainID] = []
                ChainsAndPocketsInfo["PocketResNums"][ChainID] = {}

                ChainsAndPocketsInfo["DrugScore"][ChainID] = {}
                ChainsAndPocketsInfo["PocketScore"][ChainID] = {}
                ChainsAndPocketsInfo["HydrophobicityScore"][ChainID] = {}
                ChainsAndPocketsInfo["PolarityScore"][ChainID] = {}
                ChainsAndPocketsInfo["PocketVolume"][ChainID] = {}

                ChainsAndPocketsInfo["ChainIDs"].append(ChainID)

            if PocketID not in ChainsAndPocketsInfo["PocketIDs"][ChainID]:
                ChainsAndPocketsInfo["PocketResNums"][ChainID][PocketID] = []
                ChainsAndPocketsInfo["PocketIDs"][ChainID].append(PocketID)

                # Track pocket and drug score across chains...
                ChainsAndPocketsInfo["DrugScore"][ChainID][PocketID] = DrugScore
                ChainsAndPocketsInfo["PocketScore"][ChainID][PocketID] = PocketScore
                ChainsAndPocketsInfo["HydrophobicityScore"][ChainID][PocketID] = HydrophobicityScore
                ChainsAndPocketsInfo["PolarityScore"][ChainID][PocketID] = PolarityScore
                ChainsAndPocketsInfo["PocketVolume"][ChainID][PocketID] = PocketVolume

            if ResNum not in ChainsAndPocketsInfo["PocketResNums"][ChainID][PocketID]:
                ChainsAndPocketsInfo["PocketResNums"][ChainID][PocketID].append(ResNum)

    ChainsAndPocketsInfo["ChainIDs"] = sorted(ChainsAndPocketsInfo["ChainIDs"])

    return ChainsAndPocketsInfo


def GetPolymerChainIDs(Infile, MolName):
    """Get chain IDs for main chain excluding hetero atoms."""

    pymol.cmd.load(Infile, MolName)
    ChainIDs = PyMOLUtil.GetChains(MolName)

    # Retrieve polymer chains...
    SelectedChainIDs = []
    for ChainID in ChainIDs:
        AtomsCount = pymol.cmd.count_atoms("(%s and (chain %s) and (polymer) and (not hetatm))" % (MolName, ChainID))
        if AtomsCount > 0:
            SelectedChainIDs.append(ChainID)

    pymol.cmd.delete(MolName)

    return SelectedChainIDs


def GetPocketIDs(Infile, MolName):
    """Get pocket IDs."""

    # Fpockets are annonated in PDB file as STP residue name and unique residue
    # numbers...
    pymol.cmd.load(Infile, MolName)
    SelectionCmd = "(%s and (resn STP) and hetatm)" % MolName

    pymol.stored.FpocketIDsInfo = []
    pymol.cmd.iterate(SelectionCmd, "pymol.stored.FpocketIDsInfo.append(resi)")
    pymol.cmd.delete(MolName)

    PocketIDs = []
    for PocketID in pymol.stored.FpocketIDsInfo:
        if PocketID not in PocketIDs:
            PocketIDs.append(PocketID)

    return PocketIDs


def ProcessChainAndPocketIDs():
    """Process specified chain and pocket  IDs for infiles."""

    OptionsInfo["FpocketInfilesInfo"]["SpecifiedChainsAndPocketsInfo"] = []

    for FileIndex in range(0, len(OptionsInfo["InfilesInfo"]["InfilesNames"])):
        MiscUtil.PrintInfo(
            "\nProcessing specified chain and Pocket IDs for input file %s..."
            % OptionsInfo["FpocketInfilesInfo"]["InfilesNames"][FileIndex]
        )

        ChainsAndPocketsInfo = OptionsInfo["FpocketInfilesInfo"]["ChainsAndPocketsInfo"][FileIndex]

        SpecifiedChainsAndPocketsInfo = ProcessChainsAndPocketsOptionsInfo(ChainsAndPocketsInfo)
        OptionsInfo["FpocketInfilesInfo"]["SpecifiedChainsAndPocketsInfo"].append(SpecifiedChainsAndPocketsInfo)

        CheckPresenceOfValidPocketIDs(ChainsAndPocketsInfo, SpecifiedChainsAndPocketsInfo)


def GetPocketProperties(PocketAtomFile):
    """Get pocket properites from the pocket PDB file."""

    PocketScore, DrugScore, HydrophobicityScore, PolarityScore, PocketVolume = [None] * 5

    PocketAtomFH = open(PocketAtomFile, "r")
    if PocketAtomFH is None:
        return (PocketScore, DrugScore, HydrophobicityScore, PolarityScore, PocketVolume)

    for Line in PocketAtomFH:
        Line = Line.rstrip()
        if re.search("Pocket Score", Line, re.I):
            LineWords = Line.split()
            PocketScore = LineWords[-1]
        elif re.search("Drug Score", Line, re.I):
            LineWords = Line.split()
            DrugScore = LineWords[-1]
        elif re.search("Hydrophobicity Score", Line, re.I):
            LineWords = Line.split()
            HydrophobicityScore = LineWords[-1]
        elif re.search("Polarity Score", Line, re.I):
            LineWords = Line.split()
            PolarityScore = LineWords[-1]
        elif re.search(r"Pocket volume \(Monte Carlo\)", Line, re.I):
            LineWords = Line.split()
            PocketVolume = LineWords[-1]

        if not re.match("^HEADER", Line, re.I):
            break

    PocketAtomFH.close()

    return (PocketScore, DrugScore, HydrophobicityScore, PolarityScore, PocketVolume)


def ProcessChainsAndPocketsOptionsInfo(ChainsAndPocketsInfo):
    """Process specified chain and pocket IDs using command line options."""

    SpecifiedChainsAndPocketsInfo = {}
    SpecifiedChainsAndPocketsInfo["ChainIDs"] = []
    SpecifiedChainsAndPocketsInfo["PocketIDs"] = {}

    ProcessChainsOptionInfo(ChainsAndPocketsInfo, SpecifiedChainsAndPocketsInfo)
    ProcessPocketsOptionInfo(ChainsAndPocketsInfo, SpecifiedChainsAndPocketsInfo)

    return SpecifiedChainsAndPocketsInfo


def ProcessChainsOptionInfo(ChainsAndPocketsInfo, SpecifiedChainsAndPocketsInfo):
    """Process chain IDs"""

    MiscUtil.PrintInfo("Processing chain IDs...")

    ChainsOptionName = "-c, --chainIDs"
    ChainsOptionValue = OptionsInfo["ChainIDs"]

    if re.match("^All$", ChainsOptionValue, re.I):
        SpecifiedChainsAndPocketsInfo["ChainIDs"] = ChainsAndPocketsInfo["ChainIDs"]
        return
    elif re.match("^(First|Auto)$", ChainsOptionValue, re.I):
        FirstChainID = ChainsAndPocketsInfo["ChainIDs"][0] if (len(ChainsAndPocketsInfo["ChainIDs"])) else None
        if FirstChainID is not None:
            SpecifiedChainsAndPocketsInfo["ChainIDs"].append(FirstChainID)
        return

    ChainIDs = re.sub(" ", "", ChainsOptionValue)
    if not ChainIDs:
        MiscUtil.PrintError('No valid value specified using "%s" option.' % ChainsOptionName)

    ChainIDsList = ChainsAndPocketsInfo["ChainIDs"]
    SpecifiedChainIDsList = []

    ChainIDsWords = ChainIDs.split(",")
    for ChainID in ChainIDsWords:
        if ChainID not in ChainIDsList:
            MiscUtil.PrintWarning(
                'The chain ID, %s, specified using "%s" option is not valid. It\'ll be ignored. Valid chain IDs: %s'
                % (ChainID, ChainsOptionName, ", ".join(ChainIDsList))
            )
            continue
        if ChainID in SpecifiedChainIDsList:
            MiscUtil.PrintWarning(
                'The chain ID, %s, has already been specified using "%s" option. It\'ll be ignored.'
                % (ChainID, ChainsOptionName)
            )
            continue
        SpecifiedChainIDsList.append(ChainID)

    if not len(SpecifiedChainIDsList):
        MiscUtil.PrintError(
            'No valid chain IDs "%s"  specified using "%s" option.' % (ChainsOptionValue, ChainsOptionName)
        )

    SpecifiedChainsAndPocketsInfo["ChainIDs"] = SpecifiedChainIDsList


def ProcessPocketsOptionInfo(ChainsAndPocketsInfo, SpecifiedChainsAndPocketsInfo):
    """Process pocket IDs"""

    MiscUtil.PrintInfo("Processing pocket IDs...")

    PocketsModeOptionValue = OptionsInfo["FpocketMode"]

    PocketsIDsOptionName = "--fpocketIDs"
    PocketsIDsOptionValue = OptionsInfo["FpocketIDs"]

    # Intialize pocketIDs...
    for ChainID in SpecifiedChainsAndPocketsInfo["ChainIDs"]:
        SpecifiedChainsAndPocketsInfo["PocketIDs"][ChainID] = []

    if re.match("^All$", PocketsModeOptionValue, re.I):
        for ChainID in SpecifiedChainsAndPocketsInfo["ChainIDs"]:
            SpecifiedChainsAndPocketsInfo["PocketIDs"][ChainID] = ChainsAndPocketsInfo["PocketIDs"][ChainID]
        return
    elif re.match("^TopN$", PocketsModeOptionValue, re.I):
        # Setup TopN pocket IDs for each chain...
        TopNPocketsCount = int(PocketsIDsOptionValue[0])
        for ChainID in SpecifiedChainsAndPocketsInfo["ChainIDs"]:
            TopNPocketsIDs = []
            NumOfPockets = len(ChainsAndPocketsInfo["PocketIDs"][ChainID])

            if NumOfPockets:
                if TopNPocketsCount > NumOfPockets:
                    TopNPocketsIDs = ChainsAndPocketsInfo["PocketIDs"][ChainID]
                else:
                    TopNPocketsIDs = ChainsAndPocketsInfo["PocketIDs"][ChainID][0:TopNPocketsCount]

            SpecifiedChainsAndPocketsInfo["PocketIDs"][ChainID] = TopNPocketsIDs
        return

    # Process explicitly specified pocket IDs...
    PocketIDsWords = PocketsIDsOptionValue
    if not len(PocketIDsWords):
        MiscUtil.PrintError('No valid value specified using "%s" option.' % PocketsIDsOptionName)

    for ChainID in SpecifiedChainsAndPocketsInfo["ChainIDs"]:
        PocketIDsList = ChainsAndPocketsInfo["PocketIDs"][ChainID]
        SpecifiedPocketIDsList = []

        for PocketID in PocketIDsWords:
            if PocketID not in PocketIDsList:
                MiscUtil.PrintWarning(
                    'The pocket ID, %s, specified using "%s" option is not valid for chain, %s. It\'ll be ignored. Valid pocket IDs are listed earlier.'
                    % (PocketID, PocketsIDsOptionName, ChainID)
                )
                continue

            if PocketID in SpecifiedPocketIDsList:
                MiscUtil.PrintWarning(
                    'The pocket ID, %s, has already been specified using "%s" option. It\'ll be ignored.'
                    % (PocketID, PocketsIDsOptionName)
                )
                continue

            SpecifiedPocketIDsList.append(PocketID)

        if not len(SpecifiedPocketIDsList):
            MiscUtil.PrintWarning(
                'No valid pocket IDs "%s" specified using "%s" option for chain ID, %s.'
                % (PocketsIDsOptionValue, PocketsIDsOptionName, ChainID)
            )

        SpecifiedChainsAndPocketsInfo["PocketIDs"][ChainID] = SpecifiedPocketIDsList


def CheckPresenceOfValidPocketIDs(ChainsAndPocketsInfo, SpecifiedChainsAndPocketsInfo):
    """Check presence of valid pocket IDs."""

    MiscUtil.PrintInfo("\nSpecified chain IDs: %s" % (", ".join(SpecifiedChainsAndPocketsInfo["ChainIDs"])))

    for ChainID in SpecifiedChainsAndPocketsInfo["ChainIDs"]:
        if len(SpecifiedChainsAndPocketsInfo["PocketIDs"][ChainID]):
            MiscUtil.PrintInfo(
                "Chain ID: %s; Specified PocketIDs: %s"
                % (ChainID, ", ".join(SpecifiedChainsAndPocketsInfo["PocketIDs"][ChainID]))
            )
        else:
            MiscUtil.PrintInfo("Chain IDs: %s; Specified PocketIDs: None" % (ChainID))
            MiscUtil.PrintWarning(
                "No valid pocket IDs found for chain ID, %s. PyMOL groups and objects related to fpockets won't be created."
                % (ChainID)
            )


def RetrieveFirstChainID(FileIndex, FpocketComplexMode):
    """Get first chain ID."""

    FirstChainID = None
    if FpocketComplexMode:
        ChainsAndPocketsInfo = OptionsInfo["FpocketInfilesInfo"]["ChainsAndPocketsInfo"][FileIndex]
        if len(ChainsAndPocketsInfo["ChainIDs"]):
            FirstChainID = ChainsAndPocketsInfo["ChainIDs"][0]
    else:
        ChainsAndLigandsInfo = OptionsInfo["InfilesInfo"]["ChainsAndLigandsInfo"][FileIndex]
        if len(ChainsAndLigandsInfo["ChainIDs"]):
            FirstChainID = ChainsAndLigandsInfo["ChainIDs"][0]

    return FirstChainID


def ProcessSurfaceAtomTypesColors():
    """Process surface atom types colors."""

    AtomTypesColorNamesInfo = PyMOLUtil.ProcessSurfaceAtomTypesColorsOptionsInfo(
        "--surfaceAtomTypesColors", OptionsInfo["SurfaceAtomTypesColors"]
    )
    OptionsInfo["AtomTypesColorNames"] = AtomTypesColorNamesInfo


def CheckAndSetupOutfilesDir():
    """Check and setup a directory for output files used by PyMOL."""

    Outfile = Options["--outfile"]
    OutfilesDir = Options["--outfilesDir"]

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Outfile)
    if re.match("^auto$", OutfilesDir, re.I):
        OutfilesDir = "%s_out_PyMOL" % FileName

    if os.path.isdir(OutfilesDir):
        if not OptionsInfo["Overwrite"]:
            MiscUtil.PrintError(
                'The output directory, %s, already exists. Use option "--ov" or "--overwrite" and try again.'
                % OutfilesDir
            )
        MiscUtil.PrintInfo("Using existing outout dir %s..." % OutfilesDir)
    else:
        MiscUtil.PrintInfo("Creating new output dir %s..." % OutfilesDir)
        os.mkdir(OutfilesDir)

    OptionsInfo["Outfile"] = Outfile
    OptionsInfo["OutfilesDir"] = OutfilesDir

    OptionsInfo["PMLOutfile"] = Outfile
    OptionsInfo["PMLOutfilePath"] = os.path.join(OutfilesDir, Outfile)


def ProcessOptions():
    """Process and validate command line arguments and options"""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    OptionsInfo["Align"] = True if re.match("^Yes$", Options["--align"], re.I) else False
    OptionsInfo["AlignMethod"] = Options["--alignMethod"].lower()
    OptionsInfo["AlignMode"] = Options["--alignMode"]

    OptionsInfo["FpocketPropertiesAppend"] = (
        True if re.match("^Yes$", Options["--fpocketPropertiesAppend"], re.I) else False
    )

    OptionsInfo["Infiles"] = Options["--infiles"]
    OptionsInfo["InfilesNames"] = Options["--infileNames"]

    OptionsInfo["AlignRefFile"] = Options["--alignRefFile"]
    if re.match("^FirstInputFile$", Options["--alignRefFile"], re.I):
        OptionsInfo["RefFileName"] = OptionsInfo["InfilesNames"][0]
    else:
        OptionsInfo["RefFileName"] = Options["--alignRefFile"]

    OptionsInfo["ChainIDs"] = Options["--chainIDs"]

    OptionsInfo["FpocketMode"] = Options["--fpocketMode"]
    OptionsInfo["FpocketIDs"] = Options["--fpocketIDsList"]

    OptionsInfo["Overwrite"] = Options["--overwrite"]

    OptionsInfo["LabelFontID"] = int(Options["--labelFontID"])

    OptionsInfo["PocketColorByPocketNum"] = (
        True if re.match("^Yes$", Options["--pocketColorByPocketNum"], re.I) else False
    )
    OptionsInfo["PocketLabel"] = True if re.match("^Yes$", Options["--pocketLabel"], re.I) else False
    OptionsInfo["PocketLabelType"] = Options["--pocketLabelType"]
    OptionsInfo["ThreeLetterPocketLabelType"] = (
        True if re.match("^ThreeLetter$", Options["--pocketLabelType"], re.I) else False
    )

    OptionsInfo["PocketNumOneColor"] = "gray80"

    OptionsInfo["PocketSurface"] = True if re.match("^Yes$", Options["--pocketSurface"], re.I) else False
    OptionsInfo["SurfaceChain"] = True if re.match("^Yes$", Options["--surfaceChain"], re.I) else False

    OptionsInfo["SurfaceColor"] = Options["--surfaceColor"]
    OptionsInfo["SurfaceColorPalette"] = Options["--surfaceColorPalette"]
    OptionsInfo["SurfaceAtomTypesColors"] = Options["--surfaceAtomTypesColors"]
    ProcessSurfaceAtomTypesColors()

    OptionsInfo["SphereScale"] = float(Options["--sphereScale"])
    OptionsInfo["SphereTransparency"] = float(Options["--sphereTransparency"])

    OptionsInfo["SurfaceTransparency"] = float(Options["--surfaceTransparency"])

    # Check and setup outfile dir before processing input files...
    CheckAndSetupOutfilesDir()

    RetrieveInfilesInfo()
    RetrieveRefFileInfo()
    RetrieveFpocketResultFilesInfo()

    # Process specified chain and pocket IDs..
    ProcessChainAndPocketIDs()


def RetrieveOptions():
    """Retrieve command line arguments and options"""

    # Get options...
    global Options
    Options = docopt(_docoptUsage_)

    # Set current working directory to the specified directory...
    WorkingDir = Options["--workingdir"]
    if WorkingDir:
        os.chdir(WorkingDir)

    # Handle examples option...
    if "--examples" in Options and Options["--examples"]:
        MiscUtil.PrintInfo(MiscUtil.GetExamplesTextFromDocOptText(__doc__))
        sys.exit(0)


def ValidateOptions():
    """Validate option values"""

    MiscUtil.ValidateOptionTextValue("--align", Options["--align"], "yes no")
    MiscUtil.ValidateOptionTextValue("--alignMethod", Options["--alignMethod"], "align cealign super")
    MiscUtil.ValidateOptionTextValue("--alignMode", Options["--alignMode"], "FirstChain Complex")

    MiscUtil.ValidateOptionTextValue("--fpocketPropertiesAppend", Options["--fpocketPropertiesAppend"], "yes no")

    # Expand infiles to handle presence of multiple input files...
    InfileNames = MiscUtil.ExpandFileNames(Options["--infiles"], ",")
    if not len(InfileNames):
        MiscUtil.PrintError('No input files specified for "-i, --infiles" option')

    # Validate file extensions...
    for Infile in InfileNames:
        MiscUtil.ValidateOptionFilePath("-i, --infiles", Infile)
        MiscUtil.ValidateOptionFileExt("-i, --infiles", Infile, "pdb")
        MiscUtil.ValidateOptionsDistinctFileNames("-i, --infiles", Infile, "-o, --outfile", Options["--outfile"])
    Options["--infileNames"] = InfileNames

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "pml")

    if re.match("^yes$", Options["--align"], re.I):
        if not re.match("^FirstInputFile$", Options["--alignRefFile"], re.I):
            AlignRefFile = Options["--alignRefFile"]
            MiscUtil.ValidateOptionFilePath("--alignRefFile", AlignRefFile)
            MiscUtil.ValidateOptionFileExt("--alignRefFile", AlignRefFile, "pdb")
            MiscUtil.ValidateOptionsDistinctFileNames(
                "--AlignRefFile", AlignRefFile, "-o, --outfile", Options["--outfile"]
            )

    MiscUtil.ValidateOptionTextValue("--fpocketMode", Options["--fpocketMode"], "All TopN Specify")

    FpocketIDsList = []
    if re.match("^(TopN|Specify)$", Options["--fpocketMode"], re.I):
        if Options["--fpocketIDs"] is None:
            MiscUtil.PrintError(
                'No value specified for "--fpocketIDs" during "%s" of "-f, --fpocketMode" option.'
                % (Options["--fpocketMode"])
            )

        FpocketIDs = re.sub(" ", "", Options["--fpocketIDs"])
        if not FpocketIDs:
            MiscUtil.PrintError(
                'No valid value specified for "--fpocketIDs" during "%s" of "-f, --fpocketMode" option.'
                % (Options["--fpocketMode"])
            )
        FpocketIDsWords = FpocketIDs.split(",")
        if len(FpocketIDsWords) == 0:
            MiscUtil.PrintError(
                'No valid value specified for "--fpocketIDs" during "%s" of "-f, --fpocketMode" option.'
                % (Options["--fpocketMode"])
            )

        if re.match("^TopN$", Options["--fpocketMode"], re.I):
            if len(FpocketIDsWords) > 1:
                MiscUtil.PrintError(
                    'Number of values specified for "--fpocketIDs" must be 1 during "TopN" of  "-f, --fpocketMode" option.'
                )

        for FpocketID in FpocketIDsWords:
            MiscUtil.ValidateOptionIntegerValue("--fpocketIDs", FpocketID, {">": 0})
            FpocketIDsList.append(FpocketID)

    Options["--fpocketIDsList"] = FpocketIDsList

    MiscUtil.ValidateOptionIntegerValue("--labelFontID", Options["--labelFontID"], {})

    MiscUtil.ValidateOptionTextValue("--pocketColorByPocketNum", Options["--pocketColorByPocketNum"], "yes no")
    MiscUtil.ValidateOptionTextValue("--pocketLabel", Options["--pocketLabel"], "yes no")
    MiscUtil.ValidateOptionTextValue("--pocketLabelType", Options["--pocketLabelType"], "OneLetter ThreeLetter")

    MiscUtil.ValidateOptionTextValue("--pocketSurface", Options["--pocketSurface"], "yes no")
    MiscUtil.ValidateOptionTextValue("--surfaceChain", Options["--surfaceChain"], "yes no")

    MiscUtil.ValidateOptionFloatValue("--sphereScale", Options["--sphereScale"], {">": 0.0})
    MiscUtil.ValidateOptionFloatValue("--sphereTransparency", Options["--sphereTransparency"], {">=": 0.0, "<=": 1.0})

    MiscUtil.ValidateOptionTextValue(
        "--surfaceColorPalette", Options["--surfaceColorPalette"], "RedToWhite WhiteToGreen"
    )
    MiscUtil.ValidateOptionFloatValue("--surfaceTransparency", Options["--surfaceTransparency"], {">=": 0.0, "<=": 1.0})


# Setup a usage string for docopt...
_docoptUsage_ = """
PyMOLVisualizeFpockets.py - Visualize fpockets for macromolecules.

Usage:
    PyMOLVisualizeFpockets.py [--align <yes or no>] [--alignMethod <align, cealign, super>]
                              [--alignMode <FirstChain or Complex>] [--alignRefFile <filename>]
                              [--chainIDs <First, All or ID1,ID2...>] [--fpocketMode <All, TopN, or Specify>]
                              [--fpocketIDs <Value or Value1,Value2...>] [--fpocketPropertiesAppend <yes or no>]
                              [--labelFontID <number>] [--outfilesDir <outfilesDir>] [--pocketColorByPocketNum <yes or no>]
                              [--pocketLabel <yes or no>] [--pocketLabelType <OneLetter or ThreeLetter>]
                              [--pocketSurface <yes or no>] [--sphereScale <number>] [--sphereTransparency <number>]
                              [--surfaceChain <yes or no>] [--surfaceAtomTypesColors <ColorType,ColorSpec,...>]
                              [--surfaceColor <ColorName>] [--surfaceColorPalette <RedToWhite or WhiteToGreen>]
                              [--surfaceTransparency <number>] [--overwrite] [-w <dir>] -i <infile1,infile2,infile3...> -o <outfile>
    PyMOLVisualizeFpockets.py -h | --help | -e | --examples

Description:
    Generate a PyMOL visualization file for visualizing pockets in macromolecules
    detected by an open source package named Fpocket [ Ref 166 ].

    The results of Fpocket calculations must be available in the current directory
    for all input files. A complete set of expected results is shown below:
    
        Dir: <PDBFileRoot>_out
            <PDBFileRoot>_out.pdb
                ... .. ...
            Dir: pockets
                <PDBFileRoot><PocketID>_atm.pdb
                ... .. ...
    
    The supported input file format is: PDB (.pdb)

    The supported output file formats is: PyMOL script file (.pml)

    The following directory and files are created for the visualization of pockets
    detected by Fpocket:
    
        Dir: <OutFileRoot>_out_PyMOL or <OutfilesDir>
            <OutfileRoot>.pml 
            <PDBFileRoot>.pdb
            <PDBFileRoot>_out.pdb
    
    You may visualize pockets in PyMOL by loading <OutfileRoot>.pml from
    <OutfileRoot>_out_PyMOL or <OutfilesDir> directory.

    A variety of PyMOL groups and objects may be  created for visualization of
    fpockets in macromolecules. These groups and objects correspond to complexes,
    chains, fpockets, and surfaces. A complete hierarchy of all possible PyMOL
    groups and objects is shown below:
    
        <PDBFileRoot>
            .Complex
                .Initial_PDB
                .Fpocket_PDB
            .Chain<ID>
                .Complex
                    .Complex
                .Chain
                    .Chain
                    .Surface
                        .Surface
                        .Hydrophobicity
                        .Hydrophobicity_Charge
                .FPocket<ID>
                    .FPocket
                    .Residues
                    .Surface
                        .Surface
                        .Hydrophobicity
                        .Hydrophobicity_Charge
            .Chain<ID>
                    ... ... ...
                .FPocket<ID>
                    ... ... ...
                .FPocket<ID>
                    ... ... ...
            .Chain<ID>
                ... ... ...
        <PDBFileRoot>
            .Complex
                ... ... ...
            .Chain<ID>
                ... ... ...
                .FPocket<ID>
                    ... ... ...
                .FPocket<ID>
                    ... ... ...
            .Chain<ID>
                ... ... ...

Options:
    -a, --align <yes or no>  [default: no]
        Align input files to a reference file before visualization.
    --alignMethod <align, cealign, super>  [default: super]
        Alignment methodology to use for aligning input files to a
        reference file.
    --alignMode <FirstChain or Complex>  [default: FirstChain]
        Portion of input and reference files to use for spatial alignment of
        input files against reference file.  Possible values: FirstChain or
        Complex.
        
        The FirstChain mode allows alignment of the first chain in each input
        file to the first chain in the reference file along with moving the rest
        of the complex to coordinate space of the reference file. The complete
        complex in each input file is aligned to the complete complex in reference
        file for the Complex mode.
    --alignRefFile <filename>  [default: FirstInputFile]
        Reference input file name. The default is to use the first input file
        name specified using '-i, --infiles' option.
    -c, --chainIDs <First, All or ID1,ID2...>  [default: First]
        List of chain IDs to use for visualizing fpockets in macromolecules. Possible
        values: First, All, or a comma delimited list of chain IDs. The default is to
        use the chain ID for the first chain in each input file.
    -e, --examples
        Print examples.
    -f, --fpocketMode <All, TopN, or Specify>  [default: All]
        Fpockets specification mode for visualizing fpockets across chains in
        macromolecules. Possible values: All, TopN, or specify. By default, all
        available fpockets are visualized across specified chains.
        
        The fpocket IDs must be specified using '--fpocketIDs' option for 'TopN'
        and 'Specifiy' value of '--fpocketMode '.
    --fpocketIDs <Value or Value1,Value2...>
        List of Fpocket IDs for visualizing fpockets across chains. This value is
        dependent on the value of '--fpocketMode'. The possible values are
        either a number or a comma delimited list of number for 'TopN' and 
        'Specify' value '--fpocketMode' option. For example:
        
        This option is ignored during 'All' value of '--fpocketMode' option.
    --fpocketPropertiesAppend <yes or no>  [default: yes]
        Append fpocket properties to names of PyMOL fpocket groups and
        objects. The following properties are appended to the names of PyMOL
         groups using their abbreviations and values:  PocketScore - S; DrugScore - D;
         PocketVolume - V; HydrophobicityScore - H;  PolarityScore - P.
         For example:
         
         Fpocket1_S0p50_D0p00_V638p81_Hneg6p67_P11p00.Fpocket
    -h, --help
        Print this help message.
    -i, --infiles <infile1,infile2,infile3...>
        Input PDB file names. The current directory must contain the results from
        Fpocket calculations for all the input PDB files.
    --labelFontID <number>  [default: 7]
        Font ID for drawing labels. Default: 7 (Sans Bold). Valid values: 5 to 16.
        The specified value must be a valid PyMOL font ID. No validation is
        performed. The complete lists of valid font IDs is available at:
        pymolwiki.org/index.php/Label_font_id. Examples: 5 - Sans;
        7 - Sans Bold; 9 - Serif; 10 - Serif Bold.
    -o, --outfile <outfile>
        PML output file name for visualizing fpockets. The PML outfile is created
        in a new output directory named <OutfileRoot>_out_PyMOL. In addition, the
        output directory contains all appropriate PDB files generated by Fpocket for
        visualization of fpockets in PyMOL.
    --outfilesDir <outfilesDir>  [default: auto]
        Output files directory name. Default: <OutfileRoot>_out_PyMOL.
    --pocketColorByPocketNum <yes or no>  [default: yes]
        Color fpocket residues and residue labels by pocket number. Otherwise,
        the pocket residues are colored by element names using default PyMOL
        color scheme. No color is set for residue labels.
    --pocketLabel <yes or no>  [default: yes]
        Display residue labels on fpocket residues. The residue number is always
        appended to residue label. You may specify a one or thee letter residue
        labels using '--pocketLabelType' option.
    --pocketLabelType <OneLetter or ThreeLetter>  [default: OneLetter]
        Display one or three letter residue labels on fpocket residues
    --pocketSurface <yes or no>  [default: yes]
        Surfaces around fpocket residues colored by hydrophobicity alone and
        both hydrophobicity and charge. The hydrophobicity surface is colored
        at residue level using Eisenberg hydrophobicity scale for residues and color
        gradient specified by '--surfaceColorPalette' option. The  hydrophobicity and
        charge surface is colored at atom level using colors specified for
        groups of atoms by '--surfaceAtomTypesColors' option. This scheme allows
        simultaneous mapping of hyrophobicity and charge values on the surfaces.
        
        In addition, generic surfaces colored by '--surfaceColor' are always created
        for pockets.
    --sphereScale <number>  [default: 0.4]
        Scaling factor for spheres used to display fpocket alpha spheres.
    --sphereTransparency <number>  [default: 0.25]
        Transparency for spheres used to display fpocket alpha spheres.
    --surfaceChain <yes or no>  [default: yes]
        Surfaces around individual chain colored by hydrophobicity alone and
        both hydrophobicity and charge. The hydrophobicity surface is colored
        at residue level using Eisenberg hydrophobicity scale for residues and color
        gradient specified by '--surfaceColorPalette' option. The  hydrophobicity and
        charge surface is colored at atom level using colors specified for
        groups of atoms by '--surfaceAtomTypesColors' option. This scheme allows
        simultaneous mapping of hyrophobicity and charge values on the surfaces.
        
        In addition, generic surfaces colored by '--surfaceColor' are always created
        for chains.
    --surfaceAtomTypesColors <ColorType,ColorSpec,...>  [default: auto]
        Atom colors for generating surfaces colored by hyrophobicity and charge
        around chains and pockets in proteins. It's a pairwise comma delimited list
        of atom color type and color specification for groups of atoms.
        
        The default values for color types  along wth color specifications
        are shown below: 
            
            HydrophobicAtomsColor, yellow,
            NegativelyChargedAtomsColor, red,
            PositivelyChargedAtomsColor, blue,
            OtherAtomsColor, gray90
            
        The color names must be valid PyMOL names.
        
        The color values may also be specified as space delimited RGB triplets:
             
            HydrophobicAtomsColor, 0.95 0.78 0.0,
            NegativelyChargedAtomsColor, 1.0 0.4 0.4,
            PositivelyChargedAtomsColor, 0.2 0.5 0.8,
            OtherAtomsColor, 0.95 0.95 0.95
            
    --surfaceColor <ColorName>  [default: lightblue]
        Color name for surfaces around chains and pockets. This color is not used
        for surfaces colored by hydrophobicity and charge. The color name must be
        a valid PyMOL name.
    --surfaceColorPalette <RedToWhite or WhiteToGreen>  [default: RedToWhite]
        Color palette for hydrophobic surfaces around chains and pockets in proteins.
        Possible values: RedToWhite or WhiteToGreen from most hydrophobic amino
        acid to least hydrophobic. The colors values for amino acids are taken from
        color_h script available as part of the Script Library at PyMOL Wiki.
    --surfaceTransparency <number>  [default: 0.25]
        Surface transparency for molecular surfaces.
    --overwrite
        Overwrite existing files.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To visualize all fpockets available in a directory <PDBRoot>_out for the first
    chain, along pocket residues and surfaces, in a PDB file, and generate a PML
    file in a new directory <PDBRoot>_out_pymol, type:

        % PyMOLVisualizeFpockets.py -i Sample5.pdb -o Sample5.pml

    To rerun the first example without displaying pocket residue labels,
    coloring pockets by element type, and write out a PML file, type:

        % PyMOLVisualizeFpockets.py --pocketColorByPocketNum no
          --pocketLabel no -i Sample5.pdb -o Sample5.pml

    To rerun the first example to visualize only top 5 fpockets and write out a
    PML file, type:

        % PyMOLVisualizeFpockets.py -f TopN --fpocketIDs 5 -i Sample5.pdb
          -o Sample5.pml

    To rerun the first example to visualize a specific set of fpockets and write
    out a PML file, type:

        % PyMOLVisualizeFpockets.py -f Specify --fpocketIDs "1,2,3" -i Sample5.pdb
          -o Sample5.pml

   To rerun the first example to visualize all fpockets across all chains and write
   out a PML file, type:

        % PyMOLVisualizeFpockets.py -c All -f All -i Sample5.pdb -o Sample5.pml

    To rerun the first example without displaying hydrophobic and charge surfaces
    around chain and pockets and and write out a PML file, type:

        % PyMOLVisualizeFpockets.py --surfaceChain no  --pocketSurface no
          -i Sample5.pdb -o Sample5.pml

    To visualize top 5 fpockets available in a directories <PDBRoot>_out for the
    first chain, along pocket residues and surfaces, in PDB files, aligning first
    fchain in each input file to the first chain in first input file, and generate a
    PML file in a new directory <PDBRoot>_out_pymol, type:

        % PyMOLVisualizeFpockets.py -f TopN --fpocketIDs 5 --align yes
          -i "Sample5.pdb,Sample6.pdb" -o Sample5Aligned.pml

    To visualize top 5 fpockets available in a directories <PDBRoot>_out for the
    first chain, along pocket residues and surfaces, in PDB files, aligning first
    chain in each input file to the first chain in first chain in a specified PDB
    file using a specified alignment method,, and generate a PML file in a new
    directory <PDBRoot>_out_pymol, type:

        % PyMOLVisualizeFpockets.py -f TopN --fpocketIDs 5 --align yes
          --alignMode FirstChain --alignRefFile Sample6.pdb --alignMethod super 
          -i "Sample5.pdb,Sample6.pdb" -o Sample5Aligned.pml

Author:
    Manish Sud

Collaborators:
    Joann Prescott-Roy and Pat Walters

See also:
    DownloadPDBFiles.pl, PyMOLVisualizeCavities.py,
    PyMOLVisualizeCryoEMDensity.py, PyMOLVisualizeElectronDensity.py,
    PyMOLVisualizeInterfaces.py, PyMOLVisualizeMacromolecules.py,
    PyMOLVisualizeSurfaceAndBuriedResidues.py

Copyright:
    Copyright (C) 2026 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using PyMOL, a
    molecular visualization system on an open source foundation originally
    developed by Warren DeLano.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
