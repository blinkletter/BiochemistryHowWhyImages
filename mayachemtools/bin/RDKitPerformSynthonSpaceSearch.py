#!/usr/bin/env python
#
# File: RDKitPerformSynthonSpaceSearch.py
# Author: Manish Sud <msud@san.rr.com>
#
# Acknowledgment: Dave Cosgrove
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
    from rdkit.Chem import rdSynthonSpaceSearch
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import rdRascalMCES
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

    if Options and Options["--list"]:
        # Process list option...
        ProcessListSynthonSearchSpace()
    else:
        # Process and validate command line arguments and options...
        ProcessOptions()

        # Perform actions required by the script...
        PerformSynthonSpaceSearch()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def PerformSynthonSpaceSearch():
    """Perform synthon space search."""

    Mode = OptionsInfo["Mode"]
    if re.match("^FingerprintsGeneration$", Mode, re.I):
        GenerateFingerprints()
    elif re.match("^BinaryDBFileGeneration$", Mode, re.I):
        GenerateBinaryDatabaseFile()
    elif re.match("^LibraryEnumeration$", Mode, re.I):
        PerformLibraryEnumeration()
    elif re.match("^RascalSimilaritySearch$", Mode, re.I):
        PerformRascalSimilaritySearch()
    elif re.match("^SimilaritySearch$", Mode, re.I):
        PerformSimilaritySearch()
    elif re.match("^SubstructureSearch$", Mode, re.I):
        PerformSubtructureSearch()
    else:
        MiscUtil.PrintError('The value specified, %s, for option "--mode" is not valid.' % Mode)


def GenerateFingerprints():
    """Generate fingerprints for synthons and write out a binary file."""

    MiscUtil.PrintInfo("\nGenerating fingerprints (Mode: %s)..." % OptionsInfo["Mode"])

    SynthonSpace = ReadSynthonSpaceFile(OptionsInfo["Infile"])

    StartTime = time.perf_counter()

    MiscUtil.PrintInfo("\nGenerating fingerprints (Type: %s)..." % OptionsInfo["SpecifiedFingerprints"])
    FPGenerator = InitializeFingerprintsGenerator()
    SynthonSpace.BuildSynthonFingerprints(FPGenerator)

    TotalTime = time.perf_counter() - StartTime
    MiscUtil.PrintInfo("Total time: %.2f secs" % TotalTime)

    WriteSynthonSpaceBinaryFile(SynthonSpace, OptionsInfo["Outfile"])


def GenerateBinaryDatabaseFile():
    """Write out a binary file for synthons."""

    MiscUtil.PrintInfo("\nGenerating binary database file (Mode: %s)..." % OptionsInfo["Mode"])

    SynthonSpace = ReadSynthonSpaceFile(OptionsInfo["Infile"])
    WriteSynthonSpaceBinaryFile(SynthonSpace, OptionsInfo["Outfile"])


def PerformLibraryEnumeration():
    """Enumerate library using synthons and write out a SMILES file."""

    MiscUtil.PrintInfo("\nPerforming library enumeration (Mode: %s)..." % OptionsInfo["Mode"])

    SynthonSpace = ReadSynthonSpaceFile(OptionsInfo["Infile"])

    MiscUtil.PrintInfo("\nWriting file %s ..." % OptionsInfo["Outfile"])
    SynthonSpace.WriteEnumeratedFile(OptionsInfo["Outfile"])


def PerformSimilaritySearch():
    """Perform similarity search."""

    SingleOutFileMode = OptionsInfo["SingleOutFileMode"]
    CountHitsMode = OptionsInfo["CountHitsMode"]
    SynthonSearchParams = OptionsInfo["SynthonSearchParams"]

    MiscUtil.PrintInfo(
        "\nPerforming similiarity search (Fingerprints: %s; SimilarityCutoff: %s; MaxHits: %s)..."
        % (
            OptionsInfo["SpecifiedFingerprints"],
            SynthonSearchParams["SimilarityCutoff"],
            SynthonSearchParams["MaxHits"],
        )
    )

    # Setup synthon space...
    SynthonSpace, FPGenerator = SetupSynthonSpaceForSimilaritySearch()

    # Setup out file writers...
    SingleOutFileWriter, HitsInfoWriter = SetupOutfileWriters()

    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["QueryFile"])
    QueryMols = RDKitUtil.ReadMolecules(OptionsInfo["QueryFile"], **OptionsInfo["QueryFileParams"])

    # Process query molecules...
    (QueryMolCount, ValidQueryMolCount) = [0] * 2
    for QueryMol in QueryMols:
        QueryMolCount += 1
        if QueryMol is None or RDKitUtil.IsMolEmpty(QueryMol):
            continue

        ValidQueryMolCount += 1
        QueryMolName = RDKitUtil.GetMolName(QueryMol, QueryMolCount)

        HitMols, HitMolsCount, MaxPossibleHits = PerformSynthonSpaceSimilaritySearch(
            SynthonSpace, FPGenerator, QueryMol
        )

        if CountHitsMode:
            WriteHitsInfo(HitsInfoWriter, [QueryMolName, MaxPossibleHits])
        else:
            WriteHitsInfo(HitsInfoWriter, [QueryMolName, HitMolsCount, MaxPossibleHits])

            Writer = SingleOutFileWriter if SingleOutFileMode else SetupMoleculeWriter(SingleOutFileMode, QueryMolCount)
            WriteMolecules(Writer, QueryMolName, HitMols)

            if not SingleOutFileMode:
                if Writer is not None:
                    Writer.close()

    if SingleOutFileWriter is not None:
        SingleOutFileWriter.close()

    if HitsInfoWriter is not None:
        HitsInfoWriter.close()

    MiscUtil.PrintInfo("\nTotal number of query molecules: %d" % QueryMolCount)
    MiscUtil.PrintInfo("Number of valid query  molecules: %d" % ValidQueryMolCount)
    MiscUtil.PrintInfo("Number of ignored query molecules: %d" % (QueryMolCount - ValidQueryMolCount))


def PerformSubtructureSearch():
    """Perform substructure search."""

    SingleOutFileMode = OptionsInfo["SingleOutFileMode"]
    CountHitsMode = OptionsInfo["CountHitsMode"]
    SynthonSearchParams = OptionsInfo["SynthonSearchParams"]

    MiscUtil.PrintInfo("\nPerforming substructue search (MaxHits: %s)..." % (SynthonSearchParams["MaxHits"]))

    # Setup synthon space...
    SynthonSpace = ReadSynthonSpaceFile(OptionsInfo["Infile"])

    # Setup out file writers...
    SingleOutFileWriter, HitsInfoWriter = SetupOutfileWriters()

    # Process query pattern molecules...
    MiscUtil.PrintInfo("\nProcessing query patterns...")

    QueryMolCount = 0
    for QueryMol in OptionsInfo["QueryPatternMols"]:
        QueryMolCount += 1
        QueryMolName = "Pattern%s" % QueryMolCount

        HitMols, HitMolsCount, MaxPossibleHits = PerformSynthonSpaceSubstructureSearch(SynthonSpace, QueryMol)

        if CountHitsMode:
            WriteHitsInfo(HitsInfoWriter, [QueryMolName, MaxPossibleHits])
        else:
            WriteHitsInfo(HitsInfoWriter, [QueryMolName, HitMolsCount, MaxPossibleHits])

            Writer = SingleOutFileWriter if SingleOutFileMode else SetupMoleculeWriter(SingleOutFileMode, QueryMolCount)
            WriteMolecules(Writer, QueryMolName, HitMols)

            if not SingleOutFileMode:
                if Writer is not None:
                    Writer.close()

    if SingleOutFileWriter is not None:
        SingleOutFileWriter.close()

    if HitsInfoWriter is not None:
        HitsInfoWriter.close()

    MiscUtil.PrintInfo("\nTotal number of query patterns: %d" % QueryMolCount)


def PerformRascalSimilaritySearch():
    """Perform RASCAL similarity search."""

    SingleOutFileMode = OptionsInfo["SingleOutFileMode"]
    CountHitsMode = OptionsInfo["CountHitsMode"]
    RascalSearchParams = OptionsInfo["RascalSearchParams"]
    SynthonSearchParams = OptionsInfo["SynthonSearchParams"]

    MiscUtil.PrintInfo(
        "\nPerforming RASCAL similiarity search (SimilarityThreshold: %s; MaxHits: %s)..."
        % (RascalSearchParams["SimilarityThreshold"], SynthonSearchParams["MaxHits"])
    )

    # Setup synthon space...
    SynthonSpace = ReadSynthonSpaceFile(OptionsInfo["Infile"])

    # Setup out file writers...
    SingleOutFileWriter, HitsInfoWriter = SetupOutfileWriters()

    # Setup a molecule reader...
    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["QueryFile"])
    QueryMols = RDKitUtil.ReadMolecules(OptionsInfo["QueryFile"], **OptionsInfo["QueryFileParams"])

    # Process query molecules...
    (QueryMolCount, ValidQueryMolCount) = [0] * 2
    for QueryMol in QueryMols:
        QueryMolCount += 1
        if QueryMol is None or RDKitUtil.IsMolEmpty(QueryMol):
            continue

        ValidQueryMolCount += 1
        QueryMolName = RDKitUtil.GetMolName(QueryMol, QueryMolCount)

        HitMols, HitMolsCount, MaxPossibleHits = PerformSynthonSpaceRascalSimilaritySearch(SynthonSpace, QueryMol)

        if CountHitsMode:
            WriteHitsInfo(HitsInfoWriter, [QueryMolName, MaxPossibleHits])
        else:
            WriteHitsInfo(HitsInfoWriter, [QueryMolName, HitMolsCount, MaxPossibleHits])

            Writer = SingleOutFileWriter if SingleOutFileMode else SetupMoleculeWriter(SingleOutFileMode, QueryMolCount)
            WriteMolecules(Writer, QueryMolName, HitMols)

            if not SingleOutFileMode:
                if Writer is not None:
                    Writer.close()

    if SingleOutFileWriter is not None:
        SingleOutFileWriter.close()

    if HitsInfoWriter is not None:
        HitsInfoWriter.close()

    MiscUtil.PrintInfo("\nTotal number of query molecules: %d" % QueryMolCount)
    MiscUtil.PrintInfo("Number of valid query  molecules: %d" % ValidQueryMolCount)
    MiscUtil.PrintInfo("Number of ignored query molecules: %d" % (QueryMolCount - ValidQueryMolCount))


def ProcessListSynthonSearchSpace():
    """Process list synthon search space information."""

    MiscUtil.PrintInfo("\nListing information...")

    # Validate infile..
    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "txt csv spc")

    # Process infile..
    OptionsInfo["Infile"] = Options["--infile"]

    SynthonSpace = ReadSynthonSpaceFile(OptionsInfo["Infile"])

    MiscUtil.PrintInfo("\nSummary of synthon space:\n")
    SynthonSpace.Summarise()

    ListSynthonSpaceFingerprintsType(SynthonSpace)


def PerformSynthonSpaceSimilaritySearch(SynthonSpace, FPGenerator, QueryMol):
    """Perform synthon space similarity search."""

    try:
        Results = SynthonSpace.FingerprintSearch(QueryMol, FPGenerator, params=OptionsInfo["RDKitSynthonSearchParams"])
    except Exception as ErrMsg:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("Failed to perform synthon space fingerprints seach:\n%s\n" % (ErrMsg))

    HitMols, HitMolsCount, MaxPossibleHits = GetSynthonSpaceHitMolecules(Results)

    return (HitMols, HitMolsCount, MaxPossibleHits)


def PerformSynthonSpaceRascalSimilaritySearch(SynthonSpace, QueryMol):
    """Perform synthon space RASCAL similarity search."""

    try:
        Results = SynthonSpace.RascalSearch(
            QueryMol, OptionsInfo["RDKitRascalSearchParams"], params=OptionsInfo["RDKitSynthonSearchParams"]
        )
    except Exception as ErrMsg:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("Failed to perform synthon space RASCAL similarity seach:\n%s\n" % (ErrMsg))

    HitMols, HitMolsCount, MaxPossibleHits = GetSynthonSpaceHitMolecules(Results)

    return (HitMols, HitMolsCount, MaxPossibleHits)


def PerformSynthonSpaceSubstructureSearch(SynthonSpace, QueryMol):
    """Perform synthon space substructure search."""

    try:
        Results = SynthonSpace.SubstructureSearch(
            QueryMol,
            substructMatchParams=OptionsInfo["RDKitSubstructureMatchParams"],
            params=OptionsInfo["RDKitSynthonSearchParams"],
        )
    except Exception as ErrMsg:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("Failed to perform synthon space substructure seach:\n%s\n" % (ErrMsg))

    HitMols, HitMolsCount, MaxPossibleHits = GetSynthonSpaceHitMolecules(Results)

    return (HitMols, HitMolsCount, MaxPossibleHits)


def GetSynthonSpaceHitMolecules(Results):
    """Retrieve synthon space hit molecues."""

    HitMols = Results.GetHitMolecules()

    HitMolsCount = len(HitMols)
    if HitMolsCount == 0:
        HitMols = None
        HitMolsCount = None

    MaxPossibleHits = Results.GetMaxNumResults()

    return (HitMols, HitMolsCount, MaxPossibleHits)


def SetupSynthonSpaceForSimilaritySearch():
    """Setup synthon space for similarity search."""

    SynthonSpace = ReadSynthonSpaceFile(OptionsInfo["Infile"])

    FPType, FPInfo = GetSynthonFingerprintsInfo(SynthonSpace)
    if FPType is None:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError(
            "The synthon space input file, %s, doesn't contain any fingerprints. You must specify a synthon space binary database file containing appropriate fingerprints for similarity search.."
            % OptionsInfo["Infile"]
        )

    if not re.search("%s" % OptionsInfo["SpecifiedFingerprints"], FPType, re.I):
        MiscUtil.PrintInfo("")
        MiscUtil.PrintWarning(
            'The fingerprints type, %s, in synthon space input file, %s, doesn\'t appear to match fingerprints, %s, specified using "--fingerprints" option for similarity search.'
            % (FPType, OptionsInfo["Infile"], OptionsInfo["SpecifiedFingerprints"])
        )

    FPGenerator = InitializeFingerprintsGenerator()

    return (SynthonSpace, FPGenerator)


def InitializeFingerprintsGenerator():
    """Initialize fingerprints generator."""

    FPGenerator = None
    SpecifiedFingerprints = OptionsInfo["SpecifiedFingerprints"]
    if re.match("^AtomPairs$", SpecifiedFingerprints, re.I):
        FPParamsInfo = OptionsInfo["FingerprintsParamsInfo"]["AtomPairs"]
        FPGenerator = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=FPParamsInfo["MinLength"],
            maxDistance=FPParamsInfo["MaxLength"],
            includeChirality=FPParamsInfo["UseChirality"],
            use2D=FPParamsInfo["Use2D"],
            fpSize=FPParamsInfo["FPSize"],
        )
    elif re.match("^Morgan$", SpecifiedFingerprints, re.I):
        FPParamsInfo = OptionsInfo["FingerprintsParamsInfo"]["Morgan"]
        FPGenerator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FPParamsInfo["Radius"],
            includeChirality=FPParamsInfo["UseChirality"],
            useBondTypes=FPParamsInfo["UseBondTypes"],
            includeRingMembership=FPParamsInfo["UseRingMembership"],
            fpSize=FPParamsInfo["FPSize"],
        )
    elif re.match("^MorganFeatures$", SpecifiedFingerprints, re.I):
        FPParamsInfo = OptionsInfo["FingerprintsParamsInfo"]["MorganFeatures"]
        FPGenerator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FPParamsInfo["Radius"],
            includeChirality=FPParamsInfo["UseChirality"],
            useBondTypes=FPParamsInfo["UseBondTypes"],
            includeRingMembership=FPParamsInfo["UseRingMembership"],
            fpSize=FPParamsInfo["FPSize"],
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganAtomInvGen(),
        )
    elif re.match("^PathLength$", SpecifiedFingerprints, re.I):
        FPParamsInfo = OptionsInfo["FingerprintsParamsInfo"]["PathLength"]
        FPGenerator = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=FPParamsInfo["MinPath"],
            maxPath=FPParamsInfo["MaxPath"],
            useHs=FPParamsInfo["UseExplicitHs"],
            branchedPaths=FPParamsInfo["UseBranchedPaths"],
            useBondOrder=FPParamsInfo["UseBondOrder"],
            fpSize=FPParamsInfo["FPSize"],
            numBitsPerFeature=FPParamsInfo["BitsPerHash"],
        )
    elif re.match("^TopologicalTorsions$", SpecifiedFingerprints, re.I):
        FPParamsInfo = OptionsInfo["FingerprintsParamsInfo"]["TopologicalTorsions"]
        FPGenerator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            includeChirality=FPParamsInfo["UseChirality"], fpSize=FPParamsInfo["FPSize"]
        )
    else:
        MiscUtil.PrintError('The value specified, %s, for option "--fingerprints" is not valid.')

    return FPGenerator


def ReadSynthonSpaceFile(Infile):
    """Read synthon space file."""

    MiscUtil.PrintInfo("\nReading synthon space file %s..." % Infile)
    SynthonSpace = rdSynthonSpaceSearch.SynthonSpace()

    StartTime = time.perf_counter()

    try:
        if MiscUtil.CheckFileExt(Infile, "spc"):
            SynthonSpace.ReadDBFile(Infile)
        else:
            SynthonSpace.ReadTextFile(Infile)
    except Exception as ErrMsg:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("Failed to read synthon space file:\n%s\n" % (ErrMsg))

    TotalTime = time.perf_counter() - StartTime
    MiscUtil.PrintInfo("Total time: %.2f secs" % TotalTime)

    return SynthonSpace


def WriteSynthonSpaceBinaryFile(SynthonSpace, Outfile):
    """Write synthon space binary file."""

    MiscUtil.PrintInfo("\nWriting synthon space file %s..." % Outfile)
    StartTime = time.perf_counter()

    try:
        SynthonSpace.WriteDBFile(Outfile)
    except Exception as ErrMsg:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("Failed to write synthon space file:\n%s\n" % (ErrMsg))

    TotalTime = time.perf_counter() - StartTime
    MiscUtil.PrintInfo("Total time: %.2f secs" % TotalTime)

    return SynthonSpace


def ListSynthonSpaceFingerprintsType(SynthonSpace):
    """List synthon space fingerprints type."""

    FPType, FPInfo = GetSynthonFingerprintsInfo(SynthonSpace)

    if FPInfo is None:
        MiscUtil.PrintInfo("\nFingerprints type: %s" % (FPInfo))
    else:
        MiscUtil.PrintInfo("\nFingerprints type: %s\nFingerprints Info: %s" % (FPType, FPInfo))


def GetSynthonFingerprintsInfo(SynthonSpace):
    """Get synthon fingerprints information."""

    FPInfo = SynthonSpace.GetSynthonFingerprintType()
    if len(FPInfo) == 0:
        return (None, None)

    if re.search("AtomPairArguments", FPInfo, re.I):
        FPType = "AtomPairs"
    elif re.search("MorganArguments", FPInfo, re.I):
        FPType = "Morgan or MorganFeatures"
    elif re.search("RDKitFPArguments", FPInfo, re.I):
        FPType = "PathLength"
    elif re.search("TopologicalTorsionArguments", FPInfo, re.I):
        FPType = "TopologicalTorsions"
    else:
        FPType = "Unknown"

    return (FPType, FPInfo)


def SetupMoleculeWriter(SIngleOutFile, MolCount=0):
    """Setup molecule writer."""

    TextOutFileMode = OptionsInfo["TextOutFileMode"]
    TextOutFileDelim = OptionsInfo["TextOutFileDelim"]
    TextOutFileTitleLine = OptionsInfo["TextOutFileTitleLine"]

    if SIngleOutFile:
        Outfile = OptionsInfo["Outfile"]
    else:
        Outfile = "%s_%s%s.%s" % (
            OptionsInfo["OutFileRoot"],
            OptionsInfo["OutFileSuffix"],
            MolCount,
            OptionsInfo["OutFileExt"],
        )

    if TextOutFileMode:
        Writer = open(Outfile, "w")
    else:
        Writer = RDKitUtil.MoleculesWriter(Outfile, **OptionsInfo["OutfileParams"])
    if Writer is None:
        MiscUtil.PrintError("Failed to setup a writer for output fie %s " % Outfile)

    if TextOutFileMode:
        if TextOutFileTitleLine:
            WriteTextFileHeaderLine(Writer, TextOutFileDelim)

    return Writer


def WriteTextFileHeaderLine(Writer, TextOutFileDelim):
    """Write out a header line for text files including SMILES file."""

    Line = ""
    if OptionsInfo["SubstructureSearchMode"]:
        Line = TextOutFileDelim.join(["SMILES", "Name", "QueryPatternNumber"])
    elif OptionsInfo["SimilaritySearchMode"]:
        Line = TextOutFileDelim.join(["SMILES", "Name", "Similarity", "QueryMolName"])
    elif OptionsInfo["RascalSimilaritySearchMode"]:
        Line = TextOutFileDelim.join(["SMILES", "Name", "Similarity", "QueryMolName"])

    Writer.write("%s\n" % Line)


def WriteMolecules(Writer, QueryMolName, HitMols):
    """Write hit molecules for similarity and substructure search."""

    RascalSimilaritySearchMode = OptionsInfo["RascalSimilaritySearchMode"]
    SimilaritySearchMode = OptionsInfo["SimilaritySearchMode"]
    SubstructureSearchMode = OptionsInfo["SubstructureSearchMode"]

    TextOutFileMode = OptionsInfo["TextOutFileMode"]
    TextOutFileDelim = OptionsInfo["TextOutFileDelim"]

    Compute2DCoords = OptionsInfo["OutfileParams"]["Compute2DCoords"]

    SMILESIsomeric = OptionsInfo["OutfileParams"]["SMILESIsomeric"]
    SMILESKekulize = OptionsInfo["OutfileParams"]["SMILESKekulize"]

    HitMolCount = 0
    for HitMol in HitMols:
        HitMolCount += 1

        if TextOutFileMode:
            # Write out text file including SMILES file...
            LineWords = []
            LineWords.append(Chem.MolToSmiles(HitMol, isomericSmiles=SMILESIsomeric, kekuleSmiles=SMILESKekulize))
            LineWords.append(RDKitUtil.GetMolName(HitMol, HitMolCount))

            if SimilaritySearchMode or RascalSimilaritySearchMode:
                Similarity = "%.2f" % float(HitMol.GetProp("Similarity"))
                LineWords.append(Similarity)

            LineWords.append(QueryMolName)

            Line = TextOutFileDelim.join(LineWords)
            Writer.write("%s\n" % Line)
        else:
            # Write out SD file...
            if SimilaritySearchMode or RascalSimilaritySearchMode:
                HitMol.SetProp("QueryMolName", QueryMolName)
            elif SubstructureSearchMode:
                HitMol.SetProp("QueryPatternNum", QueryMolName)

            if SimilaritySearchMode or RascalSimilaritySearchMode:
                Similarity = "%.2f" % float(HitMol.GetProp("Similarity"))
                HitMol.SetProp("Similarity", Similarity)

            if Compute2DCoords:
                AllChem.Compute2DCoords(HitMol)
            Writer.write(HitMol)


def SetupOutfileWriters():
    """Setup outfile writers."""

    SingleOutFileWriter, HitsInfoWriter = [None] * 2

    if OptionsInfo["CountHitsMode"]:
        MiscUtil.PrintInfo(
            "\nSkipping generation of output files containing hit structures and only counting hits (BuildHits: No)..."
        )
    else:
        if OptionsInfo["SingleOutFileMode"]:
            SingleOutFileWriter = SetupMoleculeWriter(OptionsInfo["SingleOutFileMode"])
            MiscUtil.PrintInfo("\nGenerating output file %s..." % OptionsInfo["Outfile"])
        else:
            MiscUtil.PrintInfo(
                "\nGenerating output file(s) %s_%s*.%s..."
                % (OptionsInfo["OutFileRoot"], OptionsInfo["OutFileSuffix"], OptionsInfo["OutFileExt"])
            )

    HitsInfoWriter = SetupHitsInfoWriter()

    return (SingleOutFileWriter, HitsInfoWriter)


def SetupHitsInfoWriter():
    """Setup hits info writer."""

    HitsInfoOutFile = OptionsInfo["HitsInfoOutFile"]
    HitsInfoOutFileDelim = OptionsInfo["HitsInfoOutFileDelim"]

    MiscUtil.PrintInfo("\nGenerating output file %s..." % HitsInfoOutFile)

    Writer = open(HitsInfoOutFile, "w")

    # Setup and write out header...
    MolIDColName = "MolID"
    if OptionsInfo["SubstructureSearchMode"]:
        MolIDColName = "QueryPatternNumber"
    elif OptionsInfo["SimilaritySearchMode"]:
        MolIDColName = "QueryMolName"
    elif OptionsInfo["RascalSimilaritySearchMode"]:
        MolIDColName = "QueryMolName"

    if OptionsInfo["CountHitsMode"]:
        Line = HitsInfoOutFileDelim.join([MolIDColName, "MaxPossibleHits"])
    else:
        Line = HitsInfoOutFileDelim.join([MolIDColName, "HitsCount", "MaxPossibleHits"])

    Writer.write("%s\n" % Line)

    return Writer


def WriteHitsInfo(Writer, HitsInfo):
    """Write hits info."""

    HitsInfoWords = ["%s" % HitInfo for HitInfo in HitsInfo]

    HitsInfoOutFileDelim = OptionsInfo["HitsInfoOutFileDelim"]
    Line = HitsInfoOutFileDelim.join(HitsInfoWords)

    Writer.write("%s\n" % Line)


def ProcessFingerprintsParameters():
    """Set up and process fingerprints parameters."""

    SetupFingerprintsNamesAndParameters()

    ProcessSpecifiedFingerprintsName()
    ProcessSpecifiedFingerprintsParameters()


def SetupFingerprintsNamesAndParameters():
    """Set up fingerprints parameters."""

    OptionsInfo["FingerprintsNames"] = ["AtomPairs", "Morgan", "MorganFeatures", "PathLength", "TopologicalTorsions"]

    OptionsInfo["FingerprintsParamsInfo"] = {}
    OptionsInfo["FingerprintsParamsInfo"]["AtomPairs"] = {
        "MinLength": 1,
        "MaxLength": 30,
        "UseChirality": False,
        "Use2D": True,
        "FPSize": 2048,
    }
    OptionsInfo["FingerprintsParamsInfo"]["Morgan"] = {
        "Radius": 2,
        "UseChirality": False,
        "UseBondTypes": True,
        "UseRingMembership": True,
        "FPSize": 2048,
    }
    OptionsInfo["FingerprintsParamsInfo"]["MorganFeatures"] = {
        "Radius": 2,
        "UseChirality": False,
        "UseBondTypes": True,
        "UseRingMembership": True,
        "FPSize": 2048,
    }
    OptionsInfo["FingerprintsParamsInfo"]["PathLength"] = {
        "MinPath": 1,
        "MaxPath": 7,
        "UseExplicitHs": True,
        "UseBranchedPaths": True,
        "UseBondOrder": True,
        "FPSize": 2048,
        "BitsPerHash": 2,
    }
    OptionsInfo["FingerprintsParamsInfo"]["TopologicalTorsions"] = {"UseChirality": False, "FPSize": 2048}


def ProcessSpecifiedFingerprintsName():
    """Process specified fingerprints name."""

    #  Set up a canonical fingerprints name map...
    CanonicalFingerprintsNamesMap = {}
    for Name in OptionsInfo["FingerprintsNames"]:
        CanonicalName = Name.lower()
        CanonicalFingerprintsNamesMap[CanonicalName] = Name

    # Validate specified fingerprints name...
    CanonicalFingerprintsName = OptionsInfo["Fingerprints"].lower()
    if CanonicalFingerprintsName not in CanonicalFingerprintsNamesMap:
        MiscUtil.PrintError(
            'The fingerprints name, %s, specified using "-f, --fingerprints" option is not a valid name.'
            % (OptionsInfo["Fingerprints"])
        )

    OptionsInfo["SpecifiedFingerprints"] = CanonicalFingerprintsNamesMap[CanonicalFingerprintsName]


def ProcessSpecifiedFingerprintsParameters():
    """Process specified fingerprints parameters."""

    if re.match("^auto$", OptionsInfo["FingerprintsParams"], re.I):
        # Nothing to process...
        return

    SpecifiedFingerprintsName = OptionsInfo["SpecifiedFingerprints"]

    # Parse specified fingerprints parameters...
    FingerprintsParams = re.sub(" ", "", OptionsInfo["FingerprintsParams"])
    if not FingerprintsParams:
        MiscUtil.PrintError(
            'No valid parameter name and value pairs specified using "--fingerprintsParams" option corrresponding to fingerprints %s.'
            % (SpecifiedFingerprintsName)
        )

    FingerprintsParamsWords = FingerprintsParams.split(",")
    if len(FingerprintsParamsWords) % 2:
        MiscUtil.PrintError(
            'The number of comma delimited paramater names and values, %d, specified using "--fingerprintsParams" option must be an even number.'
            % (len(FingerprintsParamsWords))
        )

    # Setup canonical parameter names for specified fingerprints...
    ValidParamNames = []
    CanonicalParamNamesMap = {}
    for ParamName in sorted(OptionsInfo["FingerprintsParamsInfo"][SpecifiedFingerprintsName]):
        ValidParamNames.append(ParamName)
        CanonicalParamNamesMap[ParamName.lower()] = ParamName

    # Validate and set paramater names and value...
    for Index in range(0, len(FingerprintsParamsWords), 2):
        Name = FingerprintsParamsWords[Index]
        Value = FingerprintsParamsWords[Index + 1]

        CanonicalName = Name.lower()
        if CanonicalName not in CanonicalParamNamesMap:
            MiscUtil.PrintError(
                'The parameter name, %s, specified using "--fingerprintsParams" option for fingerprints, %s, is not a valid name. Supported parameter names: %s'
                % (Name, SpecifiedFingerprintsName, " ".join(ValidParamNames))
            )

        ParamName = CanonicalParamNamesMap[CanonicalName]
        if re.match(
            "^(UseChirality|Use2D|UseBondTypes|UseRingMembership|UseExplicitHs|UseBranchedPaths|UseBondOrder)$",
            ParamName,
            re.I,
        ):
            if not re.match("^(Yes|No|True|False)$", Value, re.I):
                MiscUtil.PrintError(
                    'The parameter value, %s, specified using "--fingerprintsParams" option for fingerprints, %s, is not a valid value. Supported values: Yes No True False'
                    % (Value, SpecifiedFingerprintsName)
                )
            ParamValue = False
            if re.match("^(Yes|True)$", Value, re.I):
                ParamValue = True
        else:
            ParamValue = int(Value)
            if ParamValue <= 0:
                MiscUtil.PrintError(
                    'The parameter value, %s, specified using "--fingerprintsParams" option for fingerprints, %s, is not a valid value. Supported values: > 0'
                    % (Value, SpecifiedFingerprintsName)
                )

        # Set value...
        OptionsInfo["FingerprintsParamsInfo"][SpecifiedFingerprintsName][ParamName] = ParamValue


def ProcessOutfileParameters():
    """Process outfile related parameters"""

    Mode = OptionsInfo["Mode"]

    OptionsInfo["Outfile"] = Options["--outfile"]
    OptionsInfo["OutfileParams"] = MiscUtil.ProcessOptionOutfileParameters(
        "--outfileParams", Options["--outfileParams"], Options["--infile"], Options["--outfile"]
    )

    # OutfileMode is only used for similarity and substructure search...
    OptionsInfo["OutFileMode"] = Options["--outfileMode"]
    SingleOutFileMode = True
    if not re.match("^SingleFile$", Options["--outfileMode"], re.I):
        SingleOutFileMode = False
    OptionsInfo["SingleOutFileMode"] = SingleOutFileMode

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    OptionsInfo["OutFileRoot"] = FileName
    OptionsInfo["OutFileExt"] = FileExt

    OutFileSuffix = ""
    if re.match("^SubstructureSearch$", Mode, re.I):
        OutFileSuffix = "Pattern"
    elif re.match("^SimilaritySearch$", Mode, re.I):
        OutFileSuffix = "Mol"
    OptionsInfo["OutFileSuffix"] = OutFileSuffix

    OptionsInfo["HitsInfoOutFile"] = "%s_HitCount.csv" % OptionsInfo["OutFileRoot"]
    OptionsInfo["HitsInfoOutFileDelim"] = ","

    TextOutFileMode, TextOutFileDelim, TextOutFileTitleLine = [None] * 3
    if re.match("^(SimilaritySearch|SubstructureSearch)$", Mode, re.I):
        TextOutFileMode = False
        TextOutFileDelim = ""
        TextOutFileTitleLine = True

        if MiscUtil.CheckFileExt(Options["--outfile"], "csv"):
            TextOutFileMode = True
            TextOutFileDelim = ","
        elif MiscUtil.CheckFileExt(Options["--outfile"], "tsv txt"):
            TextOutFileMode = True
            TextOutFileDelim = "\t"
        elif MiscUtil.CheckFileExt(Options["--outfile"], "smi"):
            TextOutFileMode = True
            TextOutFileDelim = OptionsInfo["OutfileParams"]["SMILESDelimiter"]
            TextOutFileTitleLine = OptionsInfo["OutfileParams"]["SMILESTitleLine"]

    OptionsInfo["TextOutFileMode"] = TextOutFileMode
    OptionsInfo["TextOutFileDelim"] = TextOutFileDelim
    OptionsInfo["TextOutFileTitleLine"] = TextOutFileTitleLine

    if not OptionsInfo["SingleOutFileMode"]:
        FilesSpec = "%s_%s*.%s" % (OptionsInfo["OutFileRoot"], OptionsInfo["OutFileSuffix"], OptionsInfo["OutFileExt"])
        FileNames = MiscUtil.ExpandFileNames(FilesSpec)
        if len(FileNames):
            if not Options["--overwrite"]:
                MiscUtil.PrintError(
                    'The output files, %s, corresponding to output file specified, %s, for option "-o, --outfile" already exist. Use option "--ov" or "--overwrite" and try again.'
                    % (FilesSpec, OptionsInfo["Outfile"])
                )


def ProcessRascalSearchParametersOption():
    """Process option for RASCAL similarity search."""

    ParamsOptionName = "--rascalSearchParams"
    ParamsOptionValue = Options[ParamsOptionName]

    ParamsDefaultInfo = {
        "AllBestMCESs": ["bool", False],
        "CompleteAromaticRings": ["bool", True],
        "CompleteSmallestRings": ["bool", False],
        "ExactConnectionsMatch": ["bool", False],
        "IgnoreAtomAromaticity": ["bool", True],
        "IgnoreBondOrders": ["bool", False],
        "MaxBondMatchPairs": ["int", 1000],
        "MaxFragSeparation": ["int", -1],
        "MinCliqueSize": ["int", 0],
        "MinFragSize": ["int", -1],
        "ReturnEmptyMCES": ["bool", False],
        "RingMatchesRingOnly": ["bool", False],
        "SimilarityThreshold": ["float", 0.7],
        "SingleLargestFrag": ["bool", False],
        "Timeout": ["int", 60],
    }

    # Update default values to match RDKit default values...
    RDKitRascalSearchParams = rdRascalMCES.RascalOptions()
    for ParamName in ParamsDefaultInfo.keys():
        RDKitParamName = LowercaseFirstLetter(ParamName)
        if hasattr(RDKitRascalSearchParams, RDKitParamName):
            RDKitParamValue = getattr(RDKitRascalSearchParams, RDKitParamName)
            ParamsDefaultInfo[ParamName][1] = RDKitParamValue
        else:
            MiscUtil.PrintWarning(
                "The RASCAL search parameter, %s, is not available in RDKit. Ignoring parameter..." % ParamName
            )

    RascalSearchParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    for ParamName in ["MaxBondMatchPairs"]:
        ParamValue = RascalSearchParams[ParamName]
        if ParamValue <= 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )

    for ParamName in ["MinCliqueSize", "SimilarityThreshold"]:
        ParamValue = RascalSearchParams[ParamName]
        if ParamValue < 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: >= 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )
        if re.match("^SimilarityThreshold$", ParamName, re.I):
            if ParamValue > 1:
                MiscUtil.PrintError(
                    'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: <= 1\n'
                    % (ParamValue, ParamName, ParamsOptionName)
                )

    for ParamName in ["MaxFragSeparation", "MinFragSize", "Timeout"]:
        ParamValue = RascalSearchParams[ParamName]
        if not (ParamValue == -1 or ParamValue > 0):
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: -1 or > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )

    # Setup RDKit object for RASCAL match parameters...
    RDKitRascalSearchParams = rdRascalMCES.RascalOptions()
    for ParamName in RascalSearchParams.keys():
        ParamValue = RascalSearchParams[ParamName]

        # Convert first letter to lower case for RDKit param name and set its value...
        RDKitParamName = LowercaseFirstLetter(ParamName)
        if hasattr(RDKitRascalSearchParams, RDKitParamName):
            setattr(RDKitRascalSearchParams, RDKitParamName, ParamValue)
        else:
            MiscUtil.PrintWarning(
                "The RASCAL searh parameter, %s, is not available in RDKit. Ignoring parameter..." % ParamName
            )

    OptionsInfo["RascalSearchParams"] = RascalSearchParams
    OptionsInfo["RDKitRascalSearchParams"] = RDKitRascalSearchParams


def ProcessSubstructureMatchParametersOption():
    """Process option for substructure match parameters."""

    ParamsOptionName = "--substructureMatchParams"
    ParamsOptionValue = Options[ParamsOptionName]

    ParamsDefaultInfo = {
        "AromaticMatchesConjugated": ["bool", False],
        "MaxMatches": ["int", 1000],
        "MaxRecursiveMatches": ["int", 1000],
        "RecursionPossible": ["bool", True],
        "SpecifiedStereoQueryMatchesUnspecified": ["bool", False],
        "Uniquify": ["bool", True],
        "UseChirality": ["bool", False],
        "UseEnhancedStereo": ["bool", False],
        "UseGenericMatchers": ["bool", False],
    }

    # Update default values to match RDKit default values...
    RDKitSubstructureMatchParams = Chem.SubstructMatchParameters()
    for ParamName in ParamsDefaultInfo.keys():
        RDKitParamName = LowercaseFirstLetter(ParamName)
        if hasattr(RDKitSubstructureMatchParams, RDKitParamName):
            RDKitParamValue = getattr(RDKitSubstructureMatchParams, RDKitParamName)
            ParamsDefaultInfo[ParamName][1] = RDKitParamValue
        else:
            MiscUtil.PrintWarning(
                "The substructure match parameter, %s, is not available in RDKit. Ignoring parameter..." % ParamName
            )

    SubstructureMatchParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    for ParamName in ["MaxMatches", "MaxRecursiveMatches"]:
        ParamValue = SubstructureMatchParams[ParamName]
        if ParamValue <= 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )

    # Setup RDKit object for substructure match parameters...
    RDKitSubstructureMatchParams = Chem.SubstructMatchParameters()
    for ParamName in SubstructureMatchParams.keys():
        ParamValue = SubstructureMatchParams[ParamName]

        # Convert first letter to lower case for RDKit param name and set its value...
        RDKitParamName = LowercaseFirstLetter(ParamName)
        if hasattr(RDKitSubstructureMatchParams, RDKitParamName):
            setattr(RDKitSubstructureMatchParams, RDKitParamName, ParamValue)
        else:
            MiscUtil.PrintWarning(
                "The substructure match parameter, %s, is not available in RDKit. Ignoring parameter..." % ParamName
            )

    OptionsInfo["SubstructureMatchParams"] = SubstructureMatchParams
    OptionsInfo["RDKitSubstructureMatchParams"] = RDKitSubstructureMatchParams


def ProcessSynthonSearchParamatersOption():
    """Process option for synthon search parameters."""

    ParamsOptionName = "--synthonSearchParams"
    ParamsOptionValue = Options[ParamsOptionName]

    ParamsDefaultInfo = {
        "ApproxSimilarityAdjuster": ["float", 0.1],
        "BuildHits": ["bool", True],
        "FragSimilarityAdjuster": ["float", 0.1],
        "HitStart": ["int", 0],
        "MaxHits": ["int", 1000],
        "MaxNumFrags": ["int", 100000],
        "NumThreads": ["int", 1],
        "RandomSample": ["bool", False],
        "RandomSeed": ["int", -1],
        "SimilarityCutoff": ["float", 0.5],
        "TimeOut": ["int", 600],
    }

    # Update default values to match RDKit default values...
    RDKitSynthonSearchParams = rdSynthonSpaceSearch.SynthonSpaceSearchParams()
    for ParamName in ParamsDefaultInfo.keys():
        RDKitParamName = LowercaseFirstLetter(ParamName)
        if hasattr(RDKitSynthonSearchParams, RDKitParamName):
            RDKitParamValue = getattr(RDKitSynthonSearchParams, RDKitParamName)
            ParamsDefaultInfo[ParamName][1] = RDKitParamValue
        else:
            MiscUtil.PrintWarning(
                "The synthon space search paramater, %s, is not available in RDKit. Ignoring parameter..." % ParamName
            )

    SynthonSearchParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    for ParamName in ["ApproxSimilarityAdjuster", "FragSimilarityAdjuster", "SimilarityCutoff", "HitStart"]:
        ParamValue = SynthonSearchParams[ParamName]
        if ParamValue < 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: >= 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )
        if re.match("^SimilarityCutoff$", ParamName, re.I):
            if ParamValue > 1:
                MiscUtil.PrintError(
                    'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: <= 1\n'
                    % (ParamValue, ParamName, ParamsOptionName)
                )

    for ParamName in ["MaxNumFrags", "TimeOut"]:
        ParamValue = SynthonSearchParams[ParamName]
        if ParamValue <= 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )

    for ParamName in ["MaxHits", "RandomSeed"]:
        ParamValue = SynthonSearchParams[ParamName]
        if not (ParamValue == -1 or ParamValue > 0):
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: -1 or > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )

    ParamName = "NumThreads"
    ParamValue = SynthonSearchParams[ParamName]
    if ParamValue > 0:
        if ParamValue > mp.cpu_count():
            MiscUtil.PrintWarning(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is greater than number of CPUs, %s, returned by mp.cpu_count().'
                % (ParamValue, ParamName, ParamsOptionName, mp.cpu_count())
            )
    elif ParamValue < 0:
        if abs(ParamValue) > mp.cpu_count():
            MiscUtil.PrintWarning(
                'The absolute parameter value, %s, specified for parameter name, %s, using "%s" option is greater than number of CPUs, %s, returned by mp.cpu_count().'
                % (abs(ParamValue), ParamName, ParamsOptionName, mp.cpu_count())
            )

    # Setup RDKit object for synthon space search parameters...
    RDKitSynthonSearchParams = rdSynthonSpaceSearch.SynthonSpaceSearchParams()
    for ParamName in SynthonSearchParams.keys():
        ParamValue = SynthonSearchParams[ParamName]

        # Convert first letter to lower case for RDKit param name and set its value...
        RDKitParamName = LowercaseFirstLetter(ParamName)
        if hasattr(RDKitSynthonSearchParams, RDKitParamName):
            setattr(RDKitSynthonSearchParams, RDKitParamName, ParamValue)
        else:
            MiscUtil.PrintWarning(
                "The synthon space search paramater, %s, is not available in RDKit. Ignoring parameter..." % ParamName
            )

    OptionsInfo["CountHitsMode"] = False if SynthonSearchParams["BuildHits"] else True

    OptionsInfo["SynthonSearchParams"] = SynthonSearchParams
    OptionsInfo["RDKitSynthonSearchParams"] = RDKitSynthonSearchParams


def LowercaseFirstLetter(Text):
    """Convert first letter of a string to lowercase."""

    if Text is None or len(Text) == 0:
        return Text

    return Text[0].lower() + Text[1:]


def ProcessQueryPatternOption():
    """Process query pattern option."""

    QueryPattern = None if re.match("^None$", Options["--queryPattern"], re.I) else Options["--queryPattern"]
    QueryPatternMols = None

    if QueryPattern is not None:
        QueryPatternMols = []
        Patterns = QueryPattern.split()
        for Pattern in Patterns:
            PatternMol = Chem.MolFromSmarts(Pattern)
            if PatternMol is None:
                MiscUtil.PrintError(
                    'The value specified, %s, using option "--queryPattern" is not a valid SMARTS: Failed to create pattern molecule'
                    % (Pattern)
                )
            QueryPatternMols.append(PatternMol)

    OptionsInfo["QueryPattern"] = QueryPattern
    OptionsInfo["QueryPatternMols"] = QueryPatternMols


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    OptionsInfo["Mode"] = Options["--mode"]
    OptionsInfo["RascalSimilaritySearchMode"] = (
        True if re.match("^RASCALSimilaritySearch$", Options["--mode"], re.I) else False
    )
    OptionsInfo["SimilaritySearchMode"] = True if re.match("^SimilaritySearch$", Options["--mode"], re.I) else False
    OptionsInfo["SubstructureSearchMode"] = True if re.match("^SubstructureSearch$", Options["--mode"], re.I) else False

    OptionsInfo["Fingerprints"] = Options["--fingerprints"]

    OptionsInfo["FingerprintsParams"] = Options["--fingerprintsParams"]
    ProcessFingerprintsParameters()

    OptionsInfo["Infile"] = Options["--infile"]

    ProcessOutfileParameters()

    OptionsInfo["Overwrite"] = Options["--overwrite"]

    ProcessQueryPatternOption()

    OptionsInfo["QueryFile"] = None if re.match("^none$", Options["--queryFile"]) else Options["--queryFile"]
    if OptionsInfo["QueryFile"] is None:
        OptionsInfo["QueryFileParams"] = None
    else:
        OptionsInfo["QueryFileParams"] = MiscUtil.ProcessOptionInfileParameters(
            "--queryFileParams", Options["--queryFileParams"], Options["--queryFile"]
        )

    ProcessRascalSearchParametersOption()

    ProcessSubstructureMatchParametersOption()
    ProcessSynthonSearchParamatersOption()

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

    MiscUtil.ValidateOptionTextValue(
        "-m, --mode",
        Options["--mode"],
        "FingerprintsGeneration BinaryDBFileGeneration LibraryEnumeration RASCALSimilaritySearch SimilaritySearch SubstructureSearch",
    )

    MiscUtil.ValidateOptionTextValue(
        "-f, --fingerprints",
        Options["--fingerprints"],
        "AtomPairs Morgan MorganFeatures PathLength TopologicalTorsions",
    )

    MiscUtil.ValidateOptionFilePath("-i, --infile", Options["--infile"])
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "txt csv spc")

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "sdf sd smi csv tsv txt spc")
    if re.match("^SingleFile$", Options["--outfileMode"], re.I):
        MiscUtil.ValidateOptionsOutputFileOverwrite(
            "-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"]
        )
    MiscUtil.ValidateOptionsDistinctFileNames(
        "-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"]
    )

    if re.match("^(FingerprintsGeneration|BinaryDBFileGeneration)$", Options["--mode"], re.I):
        MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "spc")
        if not MiscUtil.CheckFileExt(Options["--outfile"], "spc"):
            MiscUtil.PrintError(
                'The file name specified , %s, for option "--outfile" is not valid during, %s, value of "--mode" option. Supported file formats: spc\n'
                % (Options["--outfile"], Options["--mode"])
            )
    elif re.match("^LibraryEnumeration$", Options["--mode"], re.I):
        if not MiscUtil.CheckFileExt(Options["--outfile"], "smi"):
            MiscUtil.PrintError(
                'The file name specified , %s, for option "--outfile" is not valid during, %s, value of "--mode" option. Supported file formats: smi\n'
                % (Options["--outfile"], Options["--mode"])
            )
    elif re.match("^(RASCALSimilaritySearch|SimilaritySearch|SubstructureSearch)$", Options["--mode"], re.I):
        if not MiscUtil.CheckFileExt(Options["--outfile"], "sdf sd smi csv tsv txt"):
            MiscUtil.PrintError(
                'The file name specified , %s, for option "--outfile" is not valid during, %s, value of "--mode" option. Supported file formats: sdf sd smi csv tsv txt\n'
                % (Options["--outfile"], Options["--mode"])
            )

    MiscUtil.ValidateOptionTextValue("--outfileMode", Options["--outfileMode"], "SingleFile or MultipleFiles")

    QueryPattern = Options["--queryPattern"]
    if re.match("^SubstructureSearch$", Options["--mode"], re.I):
        if re.match("^None$", QueryPattern, re.I):
            MiscUtil.PrintError(
                'You must specify a valid SMARTS pattern(s) for option "--queryPattern" during, SubstructureSearch, value of "-m, --mode" option.'
            )

    PatternMols = []
    if not re.match("^None$", QueryPattern, re.I):
        Patterns = QueryPattern.split()
        for Pattern in Patterns:
            PatternMol = Chem.MolFromSmarts(Pattern)
            if PatternMol is None:
                MiscUtil.PrintError(
                    'The value specified, %s, using option "--queryPattern" is not a valid SMARTS: Failed to create pattern molecule'
                    % (Pattern)
                )
            PatternMols.append(PatternMol)

    if re.match("^SubstructureSearch$", Options["--mode"], re.I):
        if len(PatternMols) == 0:
            MiscUtil.PrintError(
                'You must specify a valid SMARTS pattern(s) for option "--queryPattern" during, SubstructureSearch, value of "-m, --mode" option.'
            )

    if re.match("^(RASCALSimilaritySearch|SimilaritySearch)$", Options["--mode"], re.I):
        if re.match("^None$", Options["--queryFile"], re.I):
            MiscUtil.PrintError(
                'You must specify a valid filename for option "--queryFile" during, SimilaritySearch, value of "-m, --mode" option.'
            )

    if not re.match("^None$", Options["--queryFile"], re.I):
        MiscUtil.ValidateOptionFilePath("--queryFile", Options["--queryFile"])
        MiscUtil.ValidateOptionFileExt("--queryFile", Options["--queryFile"], "sdf sd smi csv tsv")


# Setup a usage string for docopt...
_docoptUsage_ = """
RDKitPerformSynthonSpaceSearch.py - Perform a synthon space search

Usage:
    RDKitPerformSynthonSpaceSearch.py [--fingerprints <Morgan, PathLength...>] [--fingerprintsParams <Name,Value,...>]
                                      [--mode <SubstructureSearch...>] [ --outfileParams <Name,Value,...>] [--outfileMode <SingleFile or MultipleFiles>]
                                      [--overwrite] [--queryPattern <SMARTS>] [--queryFileParams <Name,Value,...>] [--queryFile <filename>]
                                      [--rascalSearchParams <Name,Value,...>] [--substructureMatchParams <Name,Value,...>]
                                      [--synthonSearchParams <Name,Value,...>] [-w <dir>] -i <infile> -o <outfile>
    RDKitPerformSynthonSpaceSearch.py -l | --list -i <infile>
    RDKitPerformSynthonSpaceSearch.py -h | --help | -e | --examples

Description:
    Perform a similarity or substructure search, using query molecules or SMARTS
    patterns, against a synthon space [ Ref 174 ] in an input file, and write out the
    hit molecules to output file(s). You may optionally count the hits without
    building and writing them out.

    In addition, you may enumerate a combinatorial library corresponding to a
    synthon space, generate fingerprints for a synthon space, or list information
    about a synthon space.

    You must provide a valid synthon space text or binary database file supported
    by RDKit module rdSynthonSpaceSearch.

    You may perform similarity search using fingerprints or employ RASCAL (RApid
    Similarity CALculations using Maximum Edge Subgrahps) methodology [ Ref 175 ].

    A number of fingerprints are available for performing similarity search. The
    similarity metric, however, is calculated using Tanimoto similarity on hashed
    fingerprints. 

    The RASCAL similarity between two molecuels is calculated based on MCES
    (Maximum Common Edge Subgraphs) and corresponds to Johnson similarity.

    The supported input file formats are: CSV/TXT synthon space (.csv, .txt) or
    binary synthon space (.spc).

    The supported outfile formats, for different '--mode' values, are shown
    below:
        
        BinaryDBFileGeneration: Binary database file (.spc)
        FingerprintsGeneration: Binary database file (.spc)
        LibraryEnumeration: SMILES (.smi)
        SimilaritySearch or SubstructureSearch: SD (.sdf, .sd), SMILES (.smi),
            CSV/TSV (.csv or .tsv)
        
    Possible output files:
         
        <OutfileRoot>.<sdf,sd,smi,csv,tsv>
         
        <OutfileRoot>_Mol<Num>.<sdf,sd,smi,csv,tsv>
        <OutfileRoot>_Pattern<Num>.<sdf,sd,smi,csv,tsv>
         
         <OutfileRoot>_HitCount.csv
         
    The <OutfileRoot>_HitCount.csv contains aditional information regarding hit
     counts and is writter out for both similarity and substructure search.

Options:
    -f, --fingerprints <Morgan, PathLength...>  [default: Morgan]
        Fingerprints to use for performing synthon space similarity search.
        Supported values: AtomPairs, Morgan, MorganFeatures, PathLength,
        TopologicalTorsions. The PathLength fingerprints are Daylight like
        fingerprints. The Morgan and MorganFeature fingerprints are circular
        fingerprints, corresponding Scitegic's Extended Connectivity Fingerprints
        (ECFP) and Features Connectivity Fingerprints (FCFP). The values of
        default parameters for generating fingerprints can be modified using
        '--fingerprintsParams' option.
    --fingerprintsParams <Name,Value,...>  [default: auto]
        Parameter values to use for generating fingerprints. The default values
        are dependent on the value of '-f, --fingerprints' option. In general, it is a
        comma delimited list of parameter name and value pairs for the name of
        fingerprints specified using '-f, --fingerprints' option. The supported
        parameter names along with their default values for valid fingerprints
        names are shown below:
            
            AtomPairs: minLength,1 ,maxLength,useChirality,No,
                use2D, yes, fpSize, 2048
            Morgan: radius,2, useChirality,No, useBondTypes, yes,
                useRingMembership, yes, fpSize, 2048
            MorganFeatures: radius,2, useChirality,No, useBondTypes, yes,
                useRingMembership, yes, fpSize, 2048
            PathLength: minPath,1, maxPath,7, useExplicitHs, yes,
                useBranchedPaths, yes,useBondOrder,yes, fpSize, 2048,
                bitsPerHash,2
            TopologicalTorsions: useChirality,No, fpSize, 2048
            
        A brief description of parameters, taken from RDKit documentation, is
        provided below:
            
            AtomPairs:
            
            minLength: Minimum distance between atoms.
            maxLength: Maximum distance between atoms.
            useChirality: Use chirality for atom invariants.
            use2D: Use topological distance matrix.
            fpSize: Size of the fingerpints bit vector.
            
            Morgan and MorganFeatures:
            
            radius: Neighborhood radius.
            useChirality: Use chirality to generate fingerprints.
            useBondTypes: Use bond type for the bond invariants.
            useRingMembership: Use ring membership.
            fpSize: Size of the fingerpints bit vector.
            
            PathLength:
            
            minPath: Minimum bond path length.
            maxPath: Maximum bond path length.
            useExplicitHs: Use explicit hydrogens.
            useBranchedPaths: Use branched paths along with linear paths.
            useBondOrder: Us bond order in the path hashes.
            fpSize: Size of the fingerpints bit vector.
            bitsPerHash: Number of bits set per path.
            
            TopologicalTorsions
            
            useChirality: Use chirality to generate fingerprints.
            fpSize: Size of the fingerpints bit vector.
            
    -e, --examples
        Print examples.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Synthon space Input file name.
    -l, --list
        List information about synthon space.
    -m, --mode <SubstructureSearch...>  [default: SimilaritySearch]
        Perform similarity or substructure search, enumerate synthon space,
        or list information about a synthon space. The supported values along
        with a brief explanation of the expected behavior are shown below:
            
            BinaryDBFileGeneration: Write out a binary database file for a
                synthon space.
            FingerprintsGeneration: Generate fingerints for a synthon space and
                write out a binary database file along with fingerprints.
            LibraryEnumeration: Enumerate a combinatorial library for a synthon
                space and write out a SMILES file.
            RASCALSimilaritySearch: Perform a RASCAL (RApid Similarity
                CALculations using Maximum Edge Subgrahps) similarity search.
            SimilaritySearch: Perform a similarity search using fingerprints.
            SubstructureSearch: Perform a substructure search using specified
                SMARTS patterns.
            
    -o, --outfile <outfile>
        Output file name. The <OutfileRoot> and <OutfileExt> are used to generate
        file names during 'MultipleFiles' value for '--outfileMode' option.
    --outfileMode <SingleFile or MultipleFiles>  [default: SingleFile]
        Write out a single file containing hit molecules for substructure or
        similarity search or  generate an individual file for each query pattern
        or molecule. Possible values: SingleFile or MultipleFiles. The query
        pattern number or molecule name is written to output file(s). The query
        pattern or molecule number is also appended to output file names during
        the generation of multiple output files.
    --outfileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for writing
        molecules to files during similarity and substructue search. The supported
        parameter names for different file formats, along with their default values,
        are shown below:
            
            SD: compute2DCoords,auto,kekulize,yes,forceV3000,no
            SMILES: smilesKekulize,no,smilesDelimiter,space, smilesIsomeric,yes,
                smilesTitleLine,yes
            
        Default value for compute2DCoords: yes for SMILES input file; no for all other
        file types. The kekulize and smilesIsomeric parameters are also used during
        generation of SMILES strings for CSV/TSV files.
    --queryPattern <SMARTS SMARTS ...>  [default: none]
        A space delimited list of SMARTS patterns for performing substructure
        search. This is required for 'SubstructureSearch' value of '--mode' option.
    --queryFile <filename>  [default: none]
        Input file containing query molecules for performing similarity search. This
        is required for 'SimilaritySearch' value of '--mode' option.
    --queryFileParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for reading 
        molecules from query files during similarity search. The supported
        parameter names for different file formats, along with their default
        values, are shown below:
            
            SD, MOL: removeHydrogens,yes,sanitize,yes,strictParsing,yes
            SMILES: smilesColumn,1,smilesNameColumn,2,smilesDelimiter,space,
                smilesTitleLine,auto,sanitize,yes
            
        Possible values for smilesDelimiter: space, comma or tab.
    --rascalSearchParams <Name,Value,...>  [default: auto]
        Parameter values to use for RASCAL similarity search.
        
        The default values are automatically updated to match RDKit default values.
        The supported parameter names along with their default values are
        are shown below:
            
            allBestMCESs, no, completeAromaticRings, yes,
            completeSmallestRings, no, exactConnectionsMatch, no, 
            ignoreAtomAromaticity, yes, ignoreBondOrders, no,
            maxBondMatchPairs, 1000, maxFragSeparation, -1, minCliqueSize, 0,
            minFragSize, -1, returnEmptyMCES, false, ringMatchesRingOnly, false,
            similarityThreshold, 0.7, singleLargestFrag, no,
            timeout, 60
            
        A brief description of parameters, taken from RDKit documentation, is
        provided below:
            
            allBestMCESs: Find all Maximum Common Edge Subgraphs (MCES).
            completeAromaticRings: Use only complete aromatic rings.
            completeSmallestRings: Only complete rings present in both
                molecules.
            exactConnectionsMatch: Match atoms only when they have the same
                number of explicit connections.
            ignoreAtomAromaticity: Ignore aromaticity during atom matching.
            ignoreBondOrders: Ignore bond orders during atom matching.
            maxBondMatchPairs: Maximum number of matching bond pairs.
            maxFragSeparation: Maximum bond distance that bonds can match.
                value of -1 implies no maximum.
            minCliqueSize: A value of > 0 overrides the similarityThreshold.
                This refers to the minimum number of bonds in the MCES.
            minFragSize: Minimum number of atoms in a fragment. A value of -1
                implies no minimum.
            returnEmptyMCES: Return empty MCES results.
            ringMatchesRingOnly: Match ring bonds to only ring bonds.
            similarityThreshold: Similarity threshold for matching and
                evaluating MCES.
            singleLargestFrag: Find only a single fragment for the MCES. By
                default, multiple fragments are generated as necessary.
            timeout: Max run time in seconds. A value of -1 implies no max.
            
    --substructureMatchParams <Name,Value,...>  [default: auto]
        Parameter values to use for substructure match during synthon substructure
        search.
        
        The default values are automatically updated to match RDKit default values.
        The supported parameter names along with their default values are
        are shown below:
            
            aromaticMatchesConjugated, no, maxMatches, 1000,
            maxRecursiveMatches, 1000, recursionPossible, yes,
            specifiedStereoQueryMatchesUnspecified, no,  uniquify, yes,
            useChirality, no, useEnhancedStereo, no, useGenericMatchers, no,
            
        A brief description of parameters, taken from RDKit documentation, is
        provided below:
            
            aromaticMatchesConjugated: Match aromatic and conjugated bonds.
            maxMatches: Maximum number of matches.
            maxRecursiveMatches: Maximum number of recursive matches.
            recursionPossible: Allow recursive queries.
            specifiedStereoQueryMatchesUnspecified: Match query atoms and bonds
                with specified stereochemistry to atoms and bonds with unspecified
                stereochemistry.
            uniquify: Uniquify match results using atom indices.
            useChirality: Use chirality to match atom and bonds.
            useEnhancedStereo: Use enhanced stereochemistry during the use
                of chirality.
            useGenericMatchers: Use generic groups as a post-filtering step.
            
    --synthonSearchParams <Name,Value,...>  [default: auto]
        Parameter values to use for performing synthon substructure and similarity
        search.
        
        The default values are automatically updated to match RDKit default values.
        The supported parameter names along with their default values are
        are shown below:
            
            approxSimilarityAdjuster, 0.1, [ Default value for Morgan FPs ]
            buildHits, yes, fragSimilarityAdjuster, 0.1, hitStart, 0,
            maxHits, 1000, [ A value of -1 retrives all hits ]
            maxNumFrags, 100000,
            numThreads, 1 [ 0: Use maximum number of threads supported by the
                hardware; Negative value: Added to the maxiumum number of
                threads supported by the hardware ]
            randomSample, no,
            randomSeed, -1 [  Default value implies use random seed ]
            similarityCutoff, 0.5, [ Default for Morgan FPs. Ignored during RASCAL
                similarity search; instead, RASCAL parameter similarityThreshold is
                used.  ]
            timeOut, 600 [ Unit: sec. The RASCAL searches take longer and may
                need a higher value for timeOut. For example: 3600 ]
            
        A brief description of parameters, taken from RDKit documentation, is
        provided below:
            
            approxSimilarityAdjuster: Value used for reducing similarity cutoff
                during approximate similarity check for fingerprint search. A
                lower value leads to faster run times at the risk of missing
                some hits.
            buildHits: A no value implies to report the maximum number of hits a
                search could generate without returning any hits.
            fragSimilarityAdjuster: Value used for reducing fragment matching
                similarity cutoff to accommodate low bit densities for fragments.
            hitStart: Return hits starting from the specified sequence number
                to support retrieval of hits in batches.
            maxHits: Maximum number of hits to return. A value of -1 implies
                retrieve all hits.
            maxNumFrags: Maximum number of fragments for breaking a query. 
            numThreads: Number of threads to use for search. A value of 0 
                implies the use of all available hardware threads. A negative
                value is added to the number of available hardware threads to
                calculate number of threads to use.
            randomSample: Return a random sample of hits up to maxHits.
            randomSeed: Random number seed to use during search. A value of -1
                implies the use of a random seed.
            similarityCutoff: Similarity cutoff for returning hits by fingerprint
                similarity search. A default value of 0.5 is set for Morgan
                fingeprints.
            timeOut: Time limit for search, in seconds. A valus of  0 implies
                no timeout.
            
    --overwrite
        Overwrite existing files.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To list information about a synthon space in a text file, type:

        % RDKitPerformSynthonSpaceSearch.py --list -i SampleSynthonSpace.csv

    To generate a binary database file for a synthon space in a text file, type:

        % RDKitPerformSynthonSpaceSearch.py -m BinaryDBFileGeneration
          -i SampleSynthonSpace.csv -o SampleSynthonSpace.spc

    To enumerate a combnatorial library for a synthon space in a text file and
    write out a SMILES file, type:

        % RDKitPerformSynthonSpaceSearch.py -m LibraryEnumeration
          -i SampleSynthonSpace.csv -o SampleSynthonSpace_Library.smi

    To generate Morgan fingerprints for a synthon space in a text file, employing
    radius of 2 and bit vector size of 2048, and write out a binary database file,
    type:

        % RDKitPerformSynthonSpaceSearch.py -m FingerprintsGeneration
          -i SampleSynthonSpace.csv -o SampleSynthonSpace_MorganFPs.spc

    To perform a similarity search using Morgan fingerprints for query molecules
    in an input file, against a binary data base file synthon space containing
    Morgan fingerprints, employing radius 2 and bit vector size of 2048, finding
    a maximum of 1000 hits for each query molecule, and write out a single output
    file containing hit molecules, type:

        % RDKitPerformSynthonSpaceSearch.py -m SimilaritySearch
          -i SampleSynthonSpace_MorganFPs.spc
          --queryFile SampleSynthonSpaceQuery.sdf
          -o SampleSynthonSpace_SimilaritySearchResultsMorganFPs.sdf

    or only count hits without building hits and writing them to an output
    file:

        % RDKitPerformSynthonSpaceSearch.py -m SimilaritySearch
          -i SampleSynthonSpace_MorganFPs.spc
          --queryFile SampleSynthonSpaceQuery.sdf
          -o SampleSynthonSpace_SimilaritySearchResultsMorganFPs.sdf
          --synthonSearchParams "buildHits,No"

    To run previous example for writing individual output files for each query
    molecule, type:

        % RDKitPerformSynthonSpaceSearch.py -m SimilaritySearch
          -i SampleSynthonSpace_MorganFPs.spc
          --queryFile SampleSynthonSpaceQuery.sdf
          -o SampleSynthonSpace_SimilaritySearchResultsMorganFPs.sdf
          --outfileMode MultipleFiles

    To run previous example for retrieving all possible hits for query molecules
    and write out individual output files for each query molecules, type:

        % RDKitPerformSynthonSpaceSearch.py -m SimilaritySearch
          -i SampleSynthonSpace_MorganFPs.spc
          --queryFile SampleSynthonSpaceQuery.sdf
          -o SampleSynthonSpace_SimilaritySearchResultsMorganFPs.sdf
          --outfileMode MultipleFiles
          --synthonSearchParams "maxHits,-1"

    To run the previous example using multi-threading employing all available
    threads on your machine, retrieve maximum of 1000 hits for each query
    molecule and generate various output files, type:

        % RDKitPerformSynthonSpaceSearch.py -m SimilaritySearch
          -i SampleSynthonSpace_MorganFPs.spc
          --queryFile SampleSynthonSpaceQuery.smi
          -o SampleSynthonSpace_SimilaritySearchResultsMorganFPs.smi
          --outfileMode MultipleFiles
          --synthonSearchParams "maxHits, 1000, numThreads, 0"

    To run the previous example using multi-threading employing all but one
    available threads on your machine, type:

        % RDKitPerformSynthonSpaceSearch.py -m SimilaritySearch
          -i SampleSynthonSpace_MorganFPs.spc
          --queryFile SampleSynthonSpaceQuery.smi
          -o SampleSynthonSpace_SimilaritySearchResultsMorganFPs.smi
          --outfileMode MultipleFiles
          --synthonSearchParams "maxHits, 1000, numThreads, -1"

    To perform a substructure search using query pattern SMARTS against a synthon
    space file, finding a maximum of 1000 hits for each query pattern and write out
    a single output file containing hit molecules, type:

        % RDKitPerformSynthonSpaceSearch.py -m SubstructureSearch
          -i SampleSynthonSpace.spc
          --queryPattern "c12ccc(C)cc1[nH]nc2C(=O)NCc1cncs1"
          -o SampleSynthonSpace_SubstructureSearchResults.sdf

        % RDKitPerformSynthonSpaceSearch.py -m SubstructureSearch
          -i SampleSynthonSpace.csv
          --queryPattern 'c1c[n,s,o][n,s,o,c]c1C(=O)[$(N1CCCCC1),$(N1CCCC1)]'
          -o SampleSynthonSpace_SubstructureSearchResults.sdf

    To run previous example for retrieving for writing out individual output files
    for each query molecules, type:

        % RDKitPerformSynthonSpaceSearch.py -m SubstructureSearch
          -i SampleSynthonSpace.spc
          --queryPattern "CCN(C(=O)c1cc2cc(OC)ccc2nc1C)C1CCCN(C(=O)OC(C)(C)C)C1 
          C=CCc1c(N[C@H](C)c2cccc(C)c2)ncnc1N(C)CCCC(=O)OC"
          -o SampleSynthonSpace_SubstructureSearchResults.sdf
          --outfileMode MultipleFiles

    To perform RASCAL similarity search for query molecules in an input file,
    against a binary data base file synthon space, finding a maximum of 1000 hits
    for each query molecule, using multi-threadsing employing all available CPUs,
    timing out after 3600 seconds, and write out a single output file containing
    hit molecules, type:

        % RDKitPerformSynthonSpaceSearch.py -m RASCALSimilaritySearch
          -i SampleSynthonSpace.spc
          --queryFile SampleSynthonSpaceQuery.sdf
          -o SampleSynthonSpace_RASCALSimilaritySearchResults.sdf
          --synthonSearchParams "maxHits, 1000, numThreads, 0, timeOut, 3600"

Author:
    Manish Sud(msud@san.rr.com)

Acknowledgment:
    Dave Cosgrove

See also:
    RDKitConvertFileFormat.py, RDKitPickDiverseMolecules.py, RDKitSearchFunctionalGroups.py,
    RDKitSearchSMARTS.py

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
