#!/usr/bin/env python
#
# File: RDKitFilterChEMBLAlerts.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2026 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using RDKit, an
# open source toolkit for cheminformatics developed by Greg Landrum.
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
import multiprocessing as mp

# RDKit imports...
try:
    from rdkit import rdBase
    from rdkit import Chem
    from rdkit.Chem import AllChem
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
        "\n%s (RDKit v%s; MayaChemTools v%s; %s): Starting...\n"
        % (ScriptName, rdBase.rdkitVersion, MiscUtil.GetMayaChemToolsVersion(), time.asctime())
    )

    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()

    # Retrieve command line arguments and options...
    RetrieveOptions()

    # Process and validate command line arguments and options...
    ProcessOptions()

    # Perform actions required by the script...
    PerformFiltering()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def PerformFiltering():
    """Filter molecules using SMARTS specified in ChEMBL filters file."""

    # Setup ChEMBL patterns and pattern mols...
    MiscUtil.PrintInfo("\nSetting up ChEMBL pattern molecules for performing substructure search...")
    ChEMBLPatternMols = SetupChEMBLPatternMols()

    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols = RDKitUtil.ReadMolecules(OptionsInfo["Infile"], **OptionsInfo["InfileParams"])

    # Set up molecule writers...
    Writer, WriterFiltered = SetupMoleculeWriters()

    MolCount, ValidMolCount, RemainingMolCount = ProcessMolecules(Mols, ChEMBLPatternMols, Writer, WriterFiltered)

    if Writer is not None:
        Writer.close()
    if WriterFiltered is not None:
        WriterFiltered.close()

    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount))

    MiscUtil.PrintInfo("\nNumber of remaining molecules: %d" % RemainingMolCount)
    MiscUtil.PrintInfo("Number of filtered molecules: %d" % (ValidMolCount - RemainingMolCount))


def ProcessMolecules(Mols, ChEMBLPatternMols, Writer, WriterFiltered):
    """Process and filter molecules."""

    if OptionsInfo["MPMode"]:
        return ProcessMoleculesUsingMultipleProcesses(Mols, ChEMBLPatternMols, Writer, WriterFiltered)
    else:
        return ProcessMoleculesUsingSingleProcess(Mols, ChEMBLPatternMols, Writer, WriterFiltered)


def ProcessMoleculesUsingSingleProcess(Mols, ChEMBLPatternMols, Writer, WriterFiltered):
    """Process and filter molecules using a single process."""

    NegateMatch = OptionsInfo["NegateMatch"]
    OutfileFilteredMode = OptionsInfo["OutfileFilteredMode"]
    Compute2DCoords = OptionsInfo["OutfileParams"]["Compute2DCoords"]
    SetSMILESMolProps = OptionsInfo["OutfileParams"]["SetSMILESMolProps"]

    MiscUtil.PrintInfo("\nFiltering molecules...")

    (MolCount, ValidMolCount, RemainingMolCount) = [0] * 3
    FirstMol = True
    for Mol in Mols:
        MolCount += 1

        if Mol is None:
            continue

        if RDKitUtil.IsMolEmpty(Mol):
            MolName = RDKitUtil.GetMolName(Mol, MolCount)
            MiscUtil.PrintWarning("Ignoring empty molecule: %s" % MolName)
            continue

        ValidMolCount += 1
        if FirstMol:
            FirstMol = False
            if SetSMILESMolProps:
                SetupSMILESMoleculeWritersProps(Writer, WriterFiltered, Mol)

        MolMatched, AlertsInfo = DoesMoleculeContainsChEMBLPattern(Mol, ChEMBLPatternMols)
        if MolMatched == NegateMatch:
            RemainingMolCount += 1
            WriteMolecule(Writer, Mol, AlertsInfo, Compute2DCoords)
        else:
            if OutfileFilteredMode:
                WriteMolecule(WriterFiltered, Mol, AlertsInfo, Compute2DCoords)

    return (MolCount, ValidMolCount, RemainingMolCount)


def ProcessMoleculesUsingMultipleProcesses(Mols, ChEMBLPatternMols, Writer, WriterFiltered):
    """Process and filter molecules using multiprocessing."""

    MiscUtil.PrintInfo("\nFiltering molecules using multiprocessing...")

    MPParams = OptionsInfo["MPParams"]
    NegateMatch = OptionsInfo["NegateMatch"]
    OutfileFilteredMode = OptionsInfo["OutfileFilteredMode"]
    Compute2DCoords = OptionsInfo["OutfileParams"]["Compute2DCoords"]
    SetSMILESMolProps = OptionsInfo["OutfileParams"]["SetSMILESMolProps"]

    # Setup data for initializing a worker process...
    MiscUtil.PrintInfo("Encoding options info and ChEMBL alert pattern molecules...")
    OptionsInfo["EncodedChEMBLPatternMols"] = [
        RDKitUtil.MolToBase64EncodedMolString(PatternMol) for PatternMol in ChEMBLPatternMols
    ]
    InitializeWorkerProcessArgs = (
        MiscUtil.ObjectToBase64EncodedString(Options),
        MiscUtil.ObjectToBase64EncodedString(OptionsInfo),
    )

    # Setup a encoded mols data iterable for a worker process...
    WorkerProcessDataIterable = RDKitUtil.GenerateBase64EncodedMolStrings(Mols)

    # Setup process pool along with data initialization for each process...
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

    (MolCount, ValidMolCount, RemainingMolCount) = [0] * 3
    FirstMol = True
    for Result in Results:
        MolCount += 1
        MolIndex, EncodedMol, MolMatched, AlertsInfo = Result

        if EncodedMol is None:
            continue
        ValidMolCount += 1

        Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)

        if FirstMol:
            FirstMol = False
            if SetSMILESMolProps:
                SetupSMILESMoleculeWritersProps(Writer, WriterFiltered, Mol)

        if MolMatched == NegateMatch:
            RemainingMolCount += 1
            WriteMolecule(Writer, Mol, AlertsInfo, Compute2DCoords)
        else:
            if OutfileFilteredMode:
                WriteMolecule(WriterFiltered, Mol, AlertsInfo, Compute2DCoords)

    return (MolCount, ValidMolCount, RemainingMolCount)


def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""

    global Options, OptionsInfo

    MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())

    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Decode ChEMBLPatternMols...
    OptionsInfo["ChEMBLPatternMols"] = [
        RDKitUtil.MolFromBase64EncodedMolString(EncodedMol) for EncodedMol in OptionsInfo["EncodedChEMBLPatternMols"]
    ]


def WorkerProcess(EncodedMolInfo):
    """Process data for a worker process."""

    MolIndex, EncodedMol = EncodedMolInfo

    if EncodedMol is None:
        return [MolIndex, None, False, None]

    Mol = RDKitUtil.MolFromBase64EncodedMolString(EncodedMol)
    if RDKitUtil.IsMolEmpty(Mol):
        MolName = RDKitUtil.GetMolName(Mol, (MolIndex + 1))
        MiscUtil.PrintWarning("Ignoring empty molecule: %s" % MolName)
        return [MolIndex, None, False, None]

    MolMatched, AlertsInfo = DoesMoleculeContainsChEMBLPattern(Mol, OptionsInfo["ChEMBLPatternMols"])

    return [MolIndex, EncodedMol, MolMatched, AlertsInfo]


def WriteMolecule(Writer, Mol, AlertsInfo, Compute2DCoords):
    """Write out molecule."""

    if OptionsInfo["CountMode"]:
        return

    if Compute2DCoords:
        AllChem.Compute2DCoords(Mol)

    if AlertsInfo is not None and len(AlertsInfo):
        AlertsCount = "%s" % len(AlertsInfo)
        Alerts = "; ".join(AlertsInfo)
        if OptionsInfo["WriteAlertsCount"]:
            Mol.SetProp(OptionsInfo["AlertsCountLabel"], AlertsCount)
        Mol.SetProp(OptionsInfo["AlertsLabel"], Alerts)

    Writer.write(Mol)


def SetupMoleculeWriters():
    """Setup molecule writers."""

    Writer = None
    WriterFiltered = None

    if OptionsInfo["CountMode"]:
        return (Writer, WriterFiltered)

    Writer = RDKitUtil.MoleculesWriter(OptionsInfo["Outfile"], **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["Outfile"])
    MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["Outfile"])

    if OptionsInfo["OutfileFilteredMode"]:
        WriterFiltered = RDKitUtil.MoleculesWriter(OptionsInfo["OutfileFiltered"], **OptionsInfo["OutfileParams"])
        if WriterFiltered is None:
            MiscUtil.PrintError("Failed to setup a writer for output fie %s " % OptionsInfo["OutfileFiltered"])
        MiscUtil.PrintInfo("Generating file %s..." % OptionsInfo["OutfileFiltered"])

    return (Writer, WriterFiltered)


def SetupSMILESMoleculeWritersProps(Writer, WriterFiltered, Mol):
    """Setup properties to write for SMILES molecule writers."""

    if not OptionsInfo["OutfileParams"]["SetSMILESMolProps"]:
        return

    NegateMatch = OptionsInfo["NegateMatch"]
    SetSMILESMolAlertsProp = OptionsInfo["SetSMILESMolAlertsProp"]
    SMILESMolAlertsPropList = OptionsInfo["SMILESMolAlertsPropList"]

    if Writer is not None:
        RDKitUtil.SetWriterMolProps(Writer, Mol)
        if SetSMILESMolAlertsProp:
            if NegateMatch:
                Writer.SetProps(SMILESMolAlertsPropList)

    if WriterFiltered is not None:
        RDKitUtil.SetWriterMolProps(WriterFiltered, Mol)
        if SetSMILESMolAlertsProp:
            if not NegateMatch:
                WriterFiltered.SetProps(SMILESMolAlertsPropList)


def DoesMoleculeContainsChEMBLPattern(Mol, ChEMBLPatternMols):
    """Check presence of ChEMBL alerts pattern in the molecule."""

    MatchAllAlerts = OptionsInfo["MatchAllAlerts"]
    AlertsInfo = []
    for PatternMol in ChEMBLPatternMols:
        if Mol.HasSubstructMatch(PatternMol, useChirality=True):
            AlertsInfo.append("%s: %s" % (PatternMol.GetProp("FilterType"), PatternMol.GetProp("FilterID")))
            if not MatchAllAlerts:
                break

    if len(AlertsInfo) == 0:
        MolMatched = False
        AlertsInfo = None
    else:
        MolMatched = True

    return (MolMatched, AlertsInfo)


def SetupChEMBLPatternMols():
    """Set up ChEMBL pattern mols for substructure search corresponding to alert mode"""

    PatternMols = []
    for FilterType in OptionsInfo["SpecifiedFilterTypes"]:
        for Index, Pattern in enumerate(OptionsInfo["ChEMBLFiltersMap"]["SMARTS"][FilterType]):
            ID = OptionsInfo["ChEMBLFiltersMap"]["IDs"][FilterType][Index]

            PatternMol = Chem.MolFromSmarts(Pattern)
            if PatternMol is None:
                MiscUtil.PrintWarning("Failed to convert ChEMBL pattern, %s, into a molecule..." % Pattern)
                continue

            # Setup FilterType and PattenMol as property of PatternMol
            PatternMol.SetProp("FilterType", FilterType)
            PatternMol.SetProp("FilterID", ID)

            PatternMols.append(PatternMol)

    return PatternMols


def ProcessChEMBLAlertsMode():
    """Process specified alerts mode."""

    OptionsInfo["AlertsMode"] = Options["--alertsMode"]

    # Retrieve filetrs information...
    RetrieveChEMBLFiltersInfo()

    # Process alerts mode...
    OptionsInfo["SpecifiedFilterTypes"] = OptionsInfo["ChEMBLFiltersMap"]["FilterTypes"]
    if re.match("^All$", OptionsInfo["AlertsMode"], re.I):
        return

    AlertsMode = re.sub(" ", "", OptionsInfo["AlertsMode"])
    if not len(AlertsMode):
        MiscUtil.PrintError('The alerts mode specified using "-a, --alertsMode" option are empty.')

    CanonicalFilterTypesMap = {}
    for FilterType in OptionsInfo["ChEMBLFiltersMap"]["FilterTypes"]:
        CanonicalFilterTypesMap[FilterType.lower()] = FilterType

    SpecifiedFilterTypes = []
    for FilterType in AlertsMode.split(","):
        CanonicalFilterType = FilterType.lower()
        if CanonicalFilterType not in CanonicalFilterTypesMap:
            MiscUtil.PrintError(
                'The altert mode, %s, specified using "-a, --alertsMode" is not valid. Supported alert modes: %s'
                % (FilterType, ", ".join(OptionsInfo["ChEMBLFiltersMap"]["FilterTypes"]))
            )

        SpecifiedFilterTypes.append(CanonicalFilterTypesMap[CanonicalFilterType])

    OptionsInfo["SpecifiedFilterTypes"] = SpecifiedFilterTypes


def ProcessChEMBLAlertsMatch():
    """Process specified alerts match."""

    AlertsMatch = Options["--alertsMatch"]

    MatchFirstAlert, MatchAllAlerts = [False] * 2
    if re.match("^First$", AlertsMatch, re.I):
        MatchFirstAlert = True
    elif re.match("^All$", AlertsMatch, re.I):
        MatchAllAlerts = True
    else:
        MiscUtil.PrintError(
            'The value %s, specified using "--alertsMatch" option is not valid. Supported values: First or All'
            % (AlertsMatch)
        )

    OptionsInfo["AlertsMatch"] = AlertsMatch
    OptionsInfo["MatchFirstAlert"] = MatchFirstAlert
    OptionsInfo["MatchAllAlerts"] = MatchAllAlerts

    # Setup labels for writing out alerts match information...
    OptionsInfo["AlertsCountLabel"] = "ChEMBLAlertsCount"
    OptionsInfo["AlertsLabel"] = "FirstChEMBLAlert" if MatchFirstAlert else "ChEMBLAlerts"

    # Write out alerts count only for match all alerts...
    OptionsInfo["WriteAlertsCount"] = True if MatchAllAlerts else False

    # Write out alerts match information to comma or tab delimited SMILES files...
    SMILESDelimiter = OptionsInfo["OutfileParams"]["SMILESDelimiter"]
    OptionsInfo["SetSMILESMolAlertsProp"] = True if re.match("^[\t,]", SMILESDelimiter, re.I) else False

    SMILESMolAlertsPropList = []
    if OptionsInfo["WriteAlertsCount"]:
        SMILESMolAlertsPropList.append(OptionsInfo["AlertsCountLabel"])
    SMILESMolAlertsPropList.append(OptionsInfo["AlertsLabel"])
    OptionsInfo["SMILESMolAlertsPropList"] = SMILESMolAlertsPropList


def RetrieveChEMBLFiltersInfo():
    """Retrieve information for ChEMBL filters."""

    MayaChemToolsDataDir = MiscUtil.GetMayaChemToolsLibDataPath()
    ChEMBLFiltersFilePath = os.path.join(MayaChemToolsDataDir, "ChEMBLFilters.csv")

    MiscUtil.PrintInfo("\nRetrieving ChEMBL alerts SMARTS patterns from file %s" % (ChEMBLFiltersFilePath))

    Delimiter = ","
    QuoteChar = '"'
    IgnoreHeaderLine = True
    FilterLinesWords = MiscUtil.GetTextLinesWords(ChEMBLFiltersFilePath, Delimiter, QuoteChar, IgnoreHeaderLine)

    ChEMBLFiltersMap = {}
    ChEMBLFiltersMap["FilterTypes"] = []
    ChEMBLFiltersMap["IDs"] = {}
    ChEMBLFiltersMap["SMARTS"] = {}

    for LineWords in FilterLinesWords:
        FilterType = LineWords[0]
        ID = LineWords[1]
        SMARTS = LineWords[2]

        if FilterType not in ChEMBLFiltersMap["FilterTypes"]:
            ChEMBLFiltersMap["FilterTypes"].append(FilterType)
            ChEMBLFiltersMap["IDs"][FilterType] = []
            ChEMBLFiltersMap["SMARTS"][FilterType] = []

        ChEMBLFiltersMap["IDs"][FilterType].append(ID)
        ChEMBLFiltersMap["SMARTS"][FilterType].append(SMARTS)

    OptionsInfo["ChEMBLFiltersMap"] = ChEMBLFiltersMap

    MiscUtil.PrintInfo("\nTotal number alerts: %d" % len(FilterLinesWords))
    MiscUtil.PrintInfo(
        "Number of filter family types: %d\nFilter familty types: %s\n"
        % (len(ChEMBLFiltersMap["FilterTypes"]), ", ".join(ChEMBLFiltersMap["FilterTypes"]))
    )

    for FilterType in ChEMBLFiltersMap["FilterTypes"]:
        MiscUtil.PrintInfo(
            "Filter family type: %s; Number of alerts: %d" % (FilterType, len(ChEMBLFiltersMap["IDs"][FilterType]))
        )
    MiscUtil.PrintInfo("")


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters(
        "--infileParams", Options["--infileParams"], Options["--infile"]
    )

    OptionsInfo["Outfile"] = Options["--outfile"]
    ParamsDefaultInfoOverride = {"SMILESMolProps": True}
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters(
        "--outfileParams",
        Options["--outfileParams"],
        Options["--infile"],
        Options["--outfile"],
        ParamsDefaultInfo=ParamsDefaultInfoOverride,
    )

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    OutfileFiltered = "%s_Filtered.%s" % (FileName, FileExt)
    OptionsInfo["OutfileFiltered"] = OutfileFiltered
    OptionsInfo["OutfileFilteredMode"] = True if re.match("^yes$", Options["--outfileFiltered"], re.I) else False

    OptionsInfo["Overwrite"] = Options["--overwrite"]

    OptionsInfo["CountMode"] = True if re.match("^count$", Options["--mode"], re.I) else False
    OptionsInfo["NegateMatch"] = True if re.match("^yes$", Options["--negate"], re.I) else False

    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])

    ProcessChEMBLAlertsMode()
    ProcessChEMBLAlertsMatch()


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
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "sdf sd smi txt csv tsv")

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd smi")
    if re.match("^filter$", Options["--mode"], re.I):
        MiscUtil.ValidateOptionsOutputFileOverwrite(
            "-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"]
        )
        MiscUtil.ValidateOptionsDistinctFileNames(
            "-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"]
        )

    MiscUtil.ValidateOptionTextValue("--alertsMatch", Options["--alertsMatch"], "First All")

    MiscUtil.ValidateOptionTextValue("--outfileFiltered", Options["--outfileFiltered"], "yes no")

    MiscUtil.ValidateOptionTextValue("-m, --mode", Options["--mode"], "filter count")
    if re.match("^filter$", Options["--mode"], re.I):
        if not Options["--outfile"]:
            MiscUtil.PrintError(
                'The outfile must be specified using "-o, --outfile" during "filter" value of "-m, --mode" option'
            )

    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")
    MiscUtil.ValidateOptionTextValue("-n, --negate", Options["--negate"], "yes no")


# Setup a usage string for docopt...
_docoptUsage_ = """
RDKitFilterChEMBLAlterts.py - Filter ChEMBL alerts

Usage:
    RDKitFilterChEMBLAlerts.py  [--alertsMode <All or Type,Type,...>] [--alertsMatch <First or All>]
                                [--infileParams <Name,Value,...>] [--mode <filter or count>]
                                [--mp <yes or no>] [--mpParams <Name,Value,...>]
                                [--outfileFiltered <yes or no>] [ --outfileParams <Name,Value,...>]
                                [--negate <yes or no>] [--overwrite] [-w <dir>] -i <infile> -o <outfile>
    RDKitFilterChEMBLAlerts.py -h | --help | -e | --examples

Description:
    Filter molecules from an input file for ChEMBL structural alerts by performing
    a substructure search using SMARTS patterns specified in MAYACHEMTOOLS/
    lib/data/ChEMBLFilters.csv file and write out appropriate molecules to an
    output file or simply count the number of filtered molecules.

    The supported input file formats are: SD (.sdf, .sd), SMILES (.smi, .csv,
    .tsv, .txt)

    The supported output file formats are: SD (.sdf, .sd), SMILES (.smi)

Options:
    -a, --alertsMode <All or Type, Type,...>  [default: All]
        All or a comma delimited list of ChEMBL filter types to use for filtering
        molecules. 
        
        The supported filter family types, along with a description, are show below:
        
            BMS: Bristol-Myers Squibb HTS Deck Filters
            Dundee: University of Dundee NTD Screening Library Filters
            Glaxo: Bristol-Myers Squibb HTS Deck Filters
            Inpharmatica
            MLSMR: NIH MLSMR Excluded Functionality Filters
            PfizerLINT: Pfizer LINT filters
            SureChEMBL
        
    --alertsMatch <First or All>  [default: First]
        Stop after matching only first alert or match all ChEMBL alerts for
        filtering molecules.
        
        The 'ChEMBLAlertsCount' and 'ChEMBLAlerts' data fields are added to
        SD file containing filtered molecules for 'All' value of '-altersMatch'. In
        addition, these data fields are only written to tab or comma delimited
        SMILES file.
        
        Format:
            
            > <ChEMBLAlertsCount>
            Number
            
            > <ChEMBLAlerts>
            FilterType: ID; FilterType: ID... ... ...``
            
    -e, --examples
        Print examples.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name.
    --infileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for reading
        molecules from files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: removeHydrogens,yes,sanitize,yes,strictParsing,yes
            SMILES: smilesColumn,1,smilesNameColumn,2,smilesDelimiter,space,
                smilesTitleLine,auto,sanitize,yes
            
        Possible values for smilesDelimiter: space, comma or tab.
    -m, --mode <filter or count>  [default: filter]
        Specify whether to filter the matched molecules and write out the rest of the 
        molecules to an outfile or simply count the number of matched molecules
        marked for filtering.
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
    -n, --negate <yes or no>  [default: no]
        Specify whether to filter molecules not matching the ChEMBL filters specified by
        SMARTS patterns.
    -o, --outfile <outfile>
        Output file name.
    --outfileFiltered <yes or no>  [default: no]
        Write out a file containing filtered molecules. Its name is automatically
        generated from the specified output file. Default: <OutfileRoot>_
        Filtered.<OutfileExt>.
    --outfileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        molecules to files. The supported parameter names for different file
        formats, along with their default values, are shown below:
            
            SD: compute2DCoords,auto,kekulize,yes,forceV3000,no
            SMILES: smilesKekulize,no,smilesDelimiter,space, smilesIsomeric,yes,
                smilesTitleLine,yes,smilesMolName,yes,smilesMolProps,yes
            
        Default value for compute2DCoords: yes for SMILES input file; no for all other
        file types.
    --overwrite
        Overwrite existing files.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To count the number of molecules not containing any substructure corresponding
    to any ChEMBL SMARTS patterns and write out SMILES files containing these molecules,
    type: 

        % RDKitFilterChEMBLAlerts.py -i Sample.smi -o SampleOut.smi

    To count the number of molecules not containing any substructure corresponding
    to any ChEMBL SMARTS patterns and write out comma delmited SMILES files
    containing these and filtered molecules along with the alerts information for
    filtered molecules matching first pattern, type: 

        % RDKitFilterChEMBLAlerts.py --outfileFiltered yes --outfileParams
          "SMILESDelimiter,comma" -i Sample.smi -o SampleOut.smi

    To count the number of molecules not containing any substructure corresponding
    to any ChEMBL SMARTS patterns and write out comma delmited SMILES files
    containing these and filtered molecules along with the alerts information for
    filtered molecules matching all patterns, type: 

        % RDKitFilterChEMBLAlerts.py --alertsMatch All --outfileFiltered yes
          --outfileParams "SMILESDelimiter,comma" -i Sample.smi
          -o SampleOut.smi

    To count the number of molecules not containing any substructure corresponding
    to any ChEMBL SMARTS patterns and write out SD files containing these and filtered
    molecules along with the alerts information for filtered molecules matching all
    patterns, type: 

        % RDKitFilterChEMBLAlerts.py --alertsMatch All --outfileFiltered yes
          -i Sample.smi -o SampleOut.sdf

    To count the number of molecules not containing any substructure corresponding to
    ChEMBL SMARTS patterns, perform filtering in multiprocessing mode on all
    available CPUs without loading all data into memory, and write out a SMILES file, type: 

        % RDKitFilterChEMBLAlerts.py --mp yes -i Sample.smi -o SampleOut.smi

    To count the number of molecules not containing any substructure corresponding to
    ChEMBL SMARTS patterns, perform filtering in multiprocessing mode on all
    available CPUs by loading all data into memory, and write out a SD file, type: 

        % RDKitFilterChEMBLAlerts.py --mp yes --mpParams "inputDataMode,
          InMemory" -i Sample.smi -o SampleOut.sdf

    To count the number of molecules not containing any substructure corresponding to
    ChEMBL SMARTS patterns, perform filtering in multiprocessing mode on specific
    number of CPUs and chunk size without loading all data into memory, and
    write out a SD file, type: 

        % RDKitFilterChEMBLAlerts.py --mp yes --mpParams "inputDataMode,Lazy,
          numProcesses,4,chunkSize,8" -i Sample.smi -o SampleOut.sdf

    To only count the number of molecules not containing any substructure corresponding
    to BMS ChEMBL SMARTS patterns without writing out any files, type: 

        % RDKitFilterChEMBLAlerts.py -m count -a BMS -i Sample.sdf
          -o SampleOut.smi

    To count the number of molecules not containing any substructure corresponding
    to Pfizer LINT ChEMBL SMARTS patterns in a  CSV SMILES file and write out a SD file,
    type:  

        % RDKitFilterChEMBLAlerts.py --altertsMode PfizerLINT --infileParams
          "smilesDelimiter,comma,smilesTitleLine,yes,smilesColumn,1,
          smilesNameColumn,2" --outfileParams "compute2DCoords,yes"
          -i SampleSMILES.csv -o SampleOut.sdf

Author:
    Manish Sud(msud@san.rr.com)

See also:
    RDKitFilterPAINS.py, RDKitConvertFileFormat.py, RDKitSearchSMARTS.py

Copyright:
    Copyright (C) 2026 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using RDKit, an
    open source toolkit for cheminformatics developed by Greg Landrum.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
