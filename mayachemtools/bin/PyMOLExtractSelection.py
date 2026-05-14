#!/usr/bin/env python
#
# File: PyMOLExtractSelection.py
# Author: Manish Sud <msud@san.rr.com>
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

from __future__ import print_function

import os
import sys
import time
import re

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
        "\n%s (PyMOL v%s; MayaChemTools v%s; %s): Starting...\n"
        % (ScriptName, pymol.cmd.get_version()[0], MiscUtil.GetMayaChemToolsVersion(), time.asctime())
    )

    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()

    # Retrieve command line arguments and options...
    RetrieveOptions()

    # Process and validate command line arguments and options...
    ProcessOptions()

    # Perform actions required by the script...
    ExtractSelection()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def ExtractSelection():
    """Extract selection from input file and write it out."""

    MiscUtil.PrintInfo("\nGenerating output files...")

    # Load macromolecule from input file...
    MolName = OptionsInfo["InfileRoot"]
    pymol.cmd.load(OptionsInfo["Infile"], MolName)

    # Extract and write selection...
    ExtractAndWriteSelection(MolName)

    # Delete macromolecule...
    pymol.cmd.delete(MolName)


def ExtractAndWriteSelection(MolName):
    """Extract selection from an input file  and write it out."""

    Outfile = OptionsInfo["Outfile"]
    MiscUtil.PrintInfo("\nGenerating output file %s..." % Outfile)

    # Setup selection...
    if OptionsInfo["SelectionAppend"]:
        MolSelection = "(%s and (%s))" % (MolName, OptionsInfo["Selection"])
    else:
        MolSelection = "(%s)" % (OptionsInfo["Selection"])

    MolSelectionName = OptionsInfo["SelectionName"]

    # Create selection object and write it out...
    MiscUtil.PrintInfo("Extracting selection: %s" % MolSelection)

    pymol.cmd.create(MolSelectionName, MolSelection)
    pymol.cmd.save(Outfile, MolSelectionName)
    pymol.cmd.delete(MolSelectionName)

    if not os.path.exists(Outfile):
        MiscUtil.PrintWarning("Failed to generate output file, %s..." % (Outfile))


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["Infile"])
    OptionsInfo["InfileRoot"] = FileName

    OptionsInfo["Outfile"] = Options["--outfile"]

    OptionsInfo["Selection"] = Options["--selection"]
    OptionsInfo["SelectionName"] = "%s_Selection" % OptionsInfo["InfileRoot"]

    OptionsInfo["SelectionAppend"] = True if re.match("^Yes$", Options["--selectionAppend"], re.I) else False

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

    MiscUtil.ValidateOptionTextValue("--selectionAppend", Options["--selectionAppend"], "yes no")


# Setup a usage string for docopt...
_docoptUsage_ = """
PyMOLExtractSelection.py - Extract selection from a macromolecule

Usage:
    PyMOLExtractSelection.py [--overwrite] [--selectionAppend <yes or no>]
                             [-w <dir>] -i <infile> -o <outfile> -s <selection>
    PyMOLExtractSelection.py -h | --help | -e | --examples

Description:
    Extract data corresponding to a PyMOL selection specification from a
    macromolecule in an input file and write it out to an output file.

    The selection specification must be a valid PyMOL specification. No
    validation is performed.

    The supported input file format are:  PDB (.pdb) and CIF (.cif)

    The supported output file formats are:  PDB (.pdb) and CIF (.cif)

Options:
    -e, --examples
        Print examples.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name.
    -o, --outfile <infile>
        Output file name.
    -s, --selection <PyMOL SelectionSpec>
        Selection specification for extracting data from a macromolecule in an
        input file. The selection specification must be a valid PyMOL specification.
        No Validation is performed.
        
        The specified selection specification is optionally appended to PyMOL
        object name for input file.
    --selectionAppend <yes or no>  [default: yes]
        Append specified selection specification to  PyMOL object name for input
        file  before creating PyMOL object for a specified selection specification.
        The PyMOL object name for input file is <InfileRoot>.
        
        You may choose to explicitly specify PyMOL object name in the selection
        specification instead of automatically appending it to the selection.
    --overwrite
        Overwrite existing files.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To extract all data corresponding to chain E in a macromolecule and write
    out to a PDB file, type:

        % PyMOLExtractSelection.py -i Sample3.cif -o Sample3Out.pdb -s "chain E" --ov

    To extract only polymer chain data for chains E and I in a macromolecule and
    write out to a PDB file, type:

        % PyMOLExtractSelection.py -i Sample3.cif -o Sample3Out.pdb
          -s "((chain E) or (chain I)) and polymer" --ov

    To extract only polymer chain data for chain E in a macromolecule and write
    out to a PDB file, type:

        % PyMOLExtractSelection.py -i Sample3.pdb -o Sample3Out.pdb
          -s "(chain E) and polymer" --ov

    To extract only polymer chain data for chain E in a macromolecule by explicitly
    ignoring non-polymer chain data and write out to a CIF file, type:

        % PyMOLExtractSelection.py -i Sample3.pdb -o Sample3Out.cif
          -s "(chain E) and (not organic) and (not solvent) and
          (not inorganic)" --ov

    To extract solvent data corresponding to chain E in a macromolecule and write
    out to a PDB file, type:

        % PyMOLExtractSelection.py -i Sample3.pdb -o Sample3Out.pdb
          -s "(chain E) and solvent" --ov

    To extract ligand data corresponding to chain E in a macromolecule and write
    out to a PDB file, type:

        % PyMOLExtractSelection.py -i Sample3.pdb -o Sample3Out.pdb
          -s "(chain E) and organic" --ov

    To extract binding pocket residues with 5.0 of ligand ID ADP in chain E and write
    out a PDB file, type:

        % PyMOLExtractSelection.py -i Sample3.pdb -o Sample3Out.pdb
           --selectionAppend no -s "(byresidue (Sample3 and chain E)
          within 5.0 of (Sample3 and chain E and organic and resn ADP))
          and polymer" --ov

Author:
    Manish Sud(msud@san.rr.com)

See also:
    PyMOLAlignChains.py, PyMOLSplitChainsAndLigands.py,
    PyMOLVisualizeMacromolecules.py

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
