#!/usr/bin/env python
#
# File: OpenFECalculatePartialCharges.py
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
    import RDKitUtil
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

    # Process and validate command line arguments and options...
    ProcessOptions()

    # Perform actions required by the script...
    CalculatePartialCharges()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def CalculatePartialCharges():
    """Calculate partial charges and write them out to a SD file."""

    # Process molecules...
    Mols = ProcessMolecules()

    # Calculate charges...
    ChargedMols = CalculateCharges(Mols)

    # Write charges...
    WriteCharges(ChargedMols)


def CalculateCharges(Mols):
    """Calculate partial charges."""

    MiscUtil.PrintInfo("\nCalculating partial atomic charges (%s)..." % OptionsInfo["Charge"])

    ChargedMols, Status = OpenFEUtil.CalculatePartialCharges(Mols, OptionsInfo["Charge"], OptionsInfo["ChargeParams"])

    if not Status:
        MiscUtil.PrintError("Failed to calculate partial atomic charges.")

    return ChargedMols


def WriteCharges(ChargedMols):
    """Write charges."""

    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])
    MiscUtil.PrintInfo("\nGenerating file %s..." % OptionsInfo["Outfile"])

    RDKitChargedMols = [openfe.SmallMoleculeComponent.to_rdkit(Mol) for Mol in ChargedMols]

    for Mol in RDKitChargedMols:
        FormatPartialCharges(Mol)
        Writer.write(Mol)
    Writer.close()


def FormatPartialCharges(Mol):
    """Format partial charges."""

    Precision = OptionsInfo["ChargeParams"]["Precision"]
    LineSize = OptionsInfo["ChargeParams"]["LineSize"]

    # RDkit uses Chem.CreateAtomDoublePropertyList(rdmol, "PartialCharge") to set
    # up 'atom.dprop.PartialCharge' employing "\n" as new line delimiter.
    PropName = OpenFEUtil.GetPartialChargePropName()
    LineDelim = "\n"

    ChargesString = Mol.GetProp(PropName)

    FormattedChargesLines = []
    CurrentLine = None
    CurrentLineSize = 0

    for ChargesLine in ChargesString.split(LineDelim):
        for Value in ChargesLine.split():
            FormattedValue = "%.*f" % (Precision, float(Value))
            FormattedValueSize = len(FormattedValue)

            if (FormattedValueSize + CurrentLineSize + 1) >= LineSize:
                if CurrentLine is not None:
                    FormattedChargesLines.append(" ".join(CurrentLine))

                CurrentLine = [FormattedValue]
                CurrentLineSize = FormattedValueSize
            else:
                CurrentLineSize += FormattedValueSize
                if CurrentLine is None:
                    CurrentLine = [FormattedValue]
                else:
                    # Increment line size to account for space delimiter....
                    CurrentLineSize += 1
                    CurrentLine.append(FormattedValue)

    if CurrentLine is not None:
        FormattedChargesLines.append(" ".join(CurrentLine))

    FormattedChargesString = LineDelim.join(FormattedChargesLines)
    Mol.SetProp(PropName, FormattedChargesString)


def ProcessMolecules():
    """Process molecules."""

    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols, MolCount, ValidMolCount = OpenFEUtil.ReadAndValidateMolecules(
        OptionsInfo["InfilePath"], **OptionsInfo["InfileParams"]
    )

    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount))

    if ValidMolCount == 0:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("No valid molecules found in input file.\n")

    return Mols


def ConfigureLogging():
    """Configure logging."""

    OptionsInfo["LoggingLevel"] = Options["--loggingLevel"]

    if re.match("^Warning$", OptionsInfo["LoggingLevel"], re.I):
        LoggingLevel = logging.WARNING
    else:
        LoggingLevel = logging.INFO

    logging.basicConfig(format="%(levelname)s: %(message)s", level=LoggingLevel)


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    # Configure logging...
    ConfigureLogging()

    OptionsInfo["Infile"] = Options["--infile"]
    OptionsInfo["InfilePath"] = os.path.abspath(OptionsInfo["Infile"])

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

    OptionsInfo["Charge"] = OpenFEUtil.ProcessOptionOpenFECharge("-c, --charge", Options["--charge"])
    OptionsInfo["ChargeParams"] = OpenFEUtil.ProcessOptionOpenFEChargeParameters(
        "--chargeParams", Options["--chargeParams"], OptionsInfo["Charge"]
    )

    OptionsInfo["LoggingLevel"] = Options["--loggingLevel"]

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
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd mol")

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd")
    MiscUtil.ValidateOptionsOutputFileOverwrite(
        "-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"]
    )
    MiscUtil.ValidateOptionsDistinctFileNames(
        "-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"]
    )

    MiscUtil.ValidateOptionTextValue(
        "-c, --charge", Options["--charge"], "AM1BCC AM1-Mulliken Espaloma Gasteiger MMFF94 NAGL"
    )

    MiscUtil.ValidateOptionTextValue("--loggingLevel", Options["--loggingLevel"], "Info Warning")


# Setup a usage string for docopt...
_docoptUsage_ = """
OpenFECalculatePartialCharges.py - Generate ligand network

Usage:
    OpenFECalculatePartialCharges.py [--charge <text>]  [--chargeParams <Name,Value,..>]
                                     [--infileParams <Name,Value,...>] [--loggingLevel <Info or Warning>]
                                     [--outfileParams <Name,Value,...>] [--overwrite] [-w <dir>] -i <infile> -o <outfile>
    OpenFECalculatePartialCharges.py -h | --help | -e | --examples

Description:
    Calculate partial atomic charges for molecules in an input file and write 
    them out to a SD file.

    The partial charges are written to SD file as values of the data field label
    'atom.dprop.PartialCharge'. These values are automatically processed by
    RDKit during the loading of a SD file and assigned to atoms. You may retrieve
    these value using RDKit method Atom.GetDoubleProp('PartialCharge')

    You must specify a valid input file containing 3D coordinates for all
    molecules. In addition, the hydrogens must be present for all molecules
    in the input file.

    The supported input file formats are: Mol (.mol), SD (.sdf, .sd)

    The supported output file formats is: SD (.sdf, .sd)

Options:
    -e, --examples
        Print examples.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name.
    -c, --charge <text>  [default: AM1BCC]
        Type of partial atomic charges to calculate. Possible values: AM1BCC,
        AM1-Mulliken, Espaloma, Gasteiger, MMFF94, or NAGL.
    --chargeParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for calculating
        partial aromic charges.
        
        The supported parameter names along with their default values are
            
            naglModel, auto [ Possible value: A valid NAGL model name. By
                default, it corresponds to the latest AM1BCC production model ]
            toolkit, auto [ Possible values: RDKit or AmberTools. Default value:
                RDKit for Gasteiger and MMFF94; AmberTools for AM1BCC and
                AM1-Mulliken; Not used for Espaloma and NAGL. ]
            
            numProcessors, 1  [ Only used for AM1BCC, AM1-Mulliken, Espaloma,
                and NAGL ]
            
            precision, 4
            lineSize, 90
            
            useConformer, auto  [ Use current conformer. Possible values: yes or
                no. Default value: no for Gasteiger using AmberToolkit;
                otherwise, yes. ]
            
        A brief description of parameters is provided below:
            
            naglModel: NAGL model name. The latest AM1BCC NAGL production
                model is used by default. You must specify it explicitly in case no
                production model is available.
            toolkit: Toolkit name. RDKit for Gasteiger and MMFF94; AmberTools
                for AM1BCC, AM1-Mulliken, and Gasteiger.
            
            numProcessors: Number of processors. This is only used during the
                calculation of AM1BCC, AM1-Mulliken, Espaloma, and NAGL
                employing OpenFE method bulk_assign_partial_charges().
            
            precision: Floating point precision for writing the calculated
                partial atomic charges.
            lineSize: Line size for writing the calculated partial aromic
                charges to SD file as a string value for data field label
                'atom.dprop.PartialCharge'.
            
            useConformer: Use current conformer. The current conformer is
                always used to calculate AM1BCC, Espaloma abd NAGL charges
                using OpenFE method bulk_assign_partial_charges() and this
                option is ignored. In addition, the option value is passed to
                OpenFF method assign_partial_charges() during the calculation
                of AM1-Mulliken, Gasteiger and MMFF94 charges employing
                AmberTools or RDKit. The RDKit functions, however, ignore the
                conformer during the calculation of Gasteiger and MMFF94
                charges. The current conformer appears not used to calculate
                Gasteiger charges employing AmberTools.
            
    --infileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for reading
        molecules from files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD,MOL: removeHydrogens,no,sanitize,yes,strictParsing,yes
            
    --loggingLevel <Info or Warning>  [default: Warning]
        Logging level to configure the 'root logger' via logging.basicConfig()
        function. The default logging level is changed from 'logging.INFO' to
        'logging.WARNING'. Otherwise, OpenFE and its associated modules
        may generate a lot of informational messages.
    -o, --outfile <outfile>
        Output file name.
    --outfileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        molecules to files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: kekulize,yes,forceV3000,no
            
    --overwrite
        Overwrite existing files.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To calculate AM1BCC partial atomic charges for molecules in a SD file and
    file and write them out to a file, type:

        % OpenFECalculatePartialCharges.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsAM1BCCOut.sdf

    To run the first example for calculating AM1-Mulliken charges using
    AmberTools, type:

        % OpenFECalculatePartialCharges.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsAM1MullikenOut.sdf -c AM1-Mulliken

    To run the first example for calculating Gasteiger charges using RDKit,
    type:

        % OpenFECalculatePartialCharges.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsGasteigerRDKit.sdf -c Gasteiger

    To run the first example for calculating Gasteiger charges using
    AmberTools, type:

        % OpenFECalculatePartialCharges.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsasteigerAmberTools.sdf -c Gasteiger
          --chargeParams "toolkit,AmberTools"

    To run the first example for calculating NAGL charges using the default
    production AM1BCC model, type:

        %  OpenFECalculatePartialCharges.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsNAGL.sdf -c NAGL

    To run the first example for calculating NAGL charges using a specific
    model, type:

        %  OpenFECalculatePartialCharges.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsNAGL.sdf -c NAGL
          --chargeParams "naglmodel, openff-gnn-am1bcc-0.1.0-rc.3.pt"

    To run the first example by specifying explicit values for various parameters,
    type:

        % OpenFECalculatePartialCharges.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsAM1BCCOut.sdf -c AM1BCC
          --chargeParams "numProcessors, 4, precision, 4, lineSize, 90"
          --loggingLevel Warning

Author:
    Manish Sud(msud@san.rr.com)

See also:
    OpenFECalculateAbsoluteHydrationFreeEnergy.py, OpenFEGenerateLigandNetwork.py,
    OpenFECalculateRelativeBindingFreeEnergy.py,
    OpenFECalculateRelativeHydrationFreeEnergy.py

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
