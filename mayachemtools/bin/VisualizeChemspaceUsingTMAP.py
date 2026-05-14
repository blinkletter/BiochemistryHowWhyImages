#!/usr/bin/env python
#
# File: VisualizeChemspaceUsingTMAP.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2026 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using TMAP and
# Faerun, open source software packages for visualizing chemspace, and
# RDKit, an open source toolkit for cheminformatics developed by Greg
# Landrum.
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
import csv
import shutil
import multiprocessing as mp
import pandas as pd
import numpy as np

# TMAP and Faerun imports...
try:
    import tmap as tm
    from faerun import Faerun
    from mhfp.encoder import MHFPEncoder
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import TMAP/Faerun module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your TMAP environment and try again.\n\n")
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
    VisualizeChemspace()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def VisualizeChemspace():
    """Visualize chemspace using TMAP."""

    InfileDF = ReadMoleculeData()

    MolCount, ValidMolCount, VisualizationFailedCount = ProcessMolecules(InfileDF)

    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of molecules failed during chemspace visualization: %d" % VisualizationFailedCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount))


def ProcessMolecules(InfileDF):
    """Process molecules and generate TMAP."""

    MolCount = len(InfileDF)
    (ValidMolCount, VisualizationFailedCount) = [0] * 2

    # Setup parameter values for "auto" options based on the number of molecules...
    ProcessMolCountBasedAutoOptions(MolCount)

    # Setup LSH forest...
    LSHForest, ValidMolCount, VisualizationFailedCount = SetupLSHForest(InfileDF)
    if ValidMolCount == 0:
        return (MolCount, ValidMolCount, VisualizationFailedCount)

    SetupTMAPDisplayMessage(MolCount, ValidMolCount)

    # Generate TMAP coordinates...
    PlotCoordsInfo = GenerateTMAPCoordinates(LSHForest)

    # Setup TMAP plot data...
    PlotDataInfo = SetupTMAPPlotData(InfileDF)

    # Setup TMAP plot...
    GenerateTMAPPlot(InfileDF, PlotCoordsInfo, PlotDataInfo)

    return (MolCount, ValidMolCount, VisualizationFailedCount)


def SetupLSHForest(InfileDF):
    """Setup LSH forest."""

    if OptionsInfo["LSHForestFileRestoreMode"]:
        return RestoreLSHForest((InfileDF))
    else:
        return GenerateLSHForest(InfileDF)


def RestoreLSHForest(InfileDF):
    """Restore LSH forest."""

    (ValidMolCount, VisualizationFailedCount) = [0] * 2

    # Set valid molecule count to number of molecules in input file...
    ValidMolCount = len(InfileDF)

    LSHForestFile = OptionsInfo["OutfileLSHForest"]
    MiscUtil.PrintInfo("\nRestoring LSH forest from %s..." % LSHForestFile)
    if not os.path.isfile(LSHForestFile):
        MiscUtil.PrintError("LSH forest file %s is missing. Failed to restore LSH forest...\n" % LSHForestFile)

    LSHForest = InitializeLSHForest()
    LSHForest.restore(LSHForestFile)

    if LSHForest.size() != ValidMolCount:
        MiscUtil.PrintError(
            'The number of molecules, %s, in input file must match number of nodes, %s, in LSH forest during its restoration from a file using "--lshForestFileWrite" option.'
            % (ValidMolCount, LSHForest.size())
        )

    return (LSHForest, ValidMolCount, VisualizationFailedCount)


def GenerateLSHForest(InfileDF):
    """Generate LSH forest."""

    MinHashFingerprints, ValidMolCount, FingerprintsFailedCount = GenerateMinHashFingerprints(InfileDF)

    MiscUtil.PrintInfo("\nGenerating LSH forest...")
    LSHForest = InitializeLSHForest()

    LSHForest.batch_add(MinHashFingerprints)
    LSHForest.index()

    # Write out LSH forest...
    if OptionsInfo["LSHForestFileWriteMode"]:
        OutfileLSHForest = OptionsInfo["OutfileLSHForest"]
        if FingerprintsFailedCount > 0:
            MiscUtil.PrintWarning(
                "The MinHash fingerprints generation failed for %s molecules. Skipped writing of file %s..."
                % (FingerprintsFailedCount, OutfileLSHForest)
            )
        else:
            MiscUtil.PrintInfo("Writing LSH forest file %s..." % OutfileLSHForest)
            LSHForest.store(OutfileLSHForest)

    return (LSHForest, ValidMolCount, FingerprintsFailedCount)


def GenerateMinHashFingerprints(InfileDF):
    """Generate MinHash fingerprints."""

    if OptionsInfo["MPMode"]:
        return GenerateMinHashFingerprintsUsingMultipleProcesses(InfileDF)
    else:
        return GenerateMinHashFingerprintsUsingSingleProcess(InfileDF)


def GenerateMinHashFingerprintsUsingSingleProcess(InfileDF):
    """Generate MHFPs using a single processs."""

    MiscUtil.PrintInfo("\nGenerating MinHash fingerprints using a single process...")

    MinHashFingerprintsEncoder = InitializeMinHashFingerprintsEncoder()

    (ValidMolCount, FingerprintsFailedCount) = [0] * 2
    MinHashFingerprints = []
    FingerprintsFailedRowIndices = []

    SMILESColname = OptionsInfo["SMILESColname"]
    for MolIndex, SMILES in enumerate(InfileDF[SMILESColname]):
        MinHashFingerprint = GenerateMinHashFingerprintForMolecule(MinHashFingerprintsEncoder, SMILES)
        if MinHashFingerprint is None:
            FingerprintsFailedCount += 1
            FingerprintsFailedRowIndices.append(MolIndex)
        else:
            ValidMolCount += 1
            MinHashFingerprints.append(tm.VectorUint(MinHashFingerprint))

    # Remove failed molecules from the dataframe...
    RemoveFingerprintsFailedRows(InfileDF, FingerprintsFailedRowIndices)

    return (MinHashFingerprints, ValidMolCount, FingerprintsFailedCount)


def GenerateMinHashFingerprintsUsingMultipleProcesses(InfileDF):
    """Generate MHFPs using multiprocessing."""

    MiscUtil.PrintInfo("\nGenerating MinHash fingerprints using multiprocessing...")

    MPParams = OptionsInfo["MPParams"]

    # Setup data for initializing a worker process...
    InitializeWorkerProcessArgs = (
        MiscUtil.ObjectToBase64EncodedString(Options),
        MiscUtil.ObjectToBase64EncodedString(OptionsInfo),
    )

    # Setup SMILES iterator...
    SMILESColname = OptionsInfo["SMILESColname"]
    WorkerProcessDataIterable = SetupSMILESWithMolIndices(InfileDF[SMILESColname])

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

    (ValidMolCount, FingerprintsFailedCount) = [0] * 2
    MinHashFingerprints = []
    FingerprintsFailedRowIndices = []

    for Result in Results:
        Molndex, MinHashFingerprint = Result

        if MinHashFingerprint is None:
            FingerprintsFailedCount += 1
            FingerprintsFailedRowIndices.append(Molndex)
        else:
            ValidMolCount += 1
            MinHashFingerprints.append(tm.VectorUint(np.array(MinHashFingerprint)))

    # Remove failed molecules from the dataframe...
    RemoveFingerprintsFailedRows(InfileDF, FingerprintsFailedRowIndices)

    return (MinHashFingerprints, ValidMolCount, FingerprintsFailedCount)


def InitializeWorkerProcess(*EncodedArgs):
    """Initialize data for a worker process."""

    global Options, OptionsInfo

    if not OptionsInfo["QuietMode"]:
        MiscUtil.PrintInfo("Starting process (PID: %s)..." % os.getpid())

    # Decode Options and OptionInfo...
    Options = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[0])
    OptionsInfo = MiscUtil.ObjectFromBase64EncodedString(EncodedArgs[1])

    # Initialize MHFP encoder...
    OptionsInfo["MinHashFingerprintsEncoder"] = InitializeMinHashFingerprintsEncoder()


def WorkerProcess(MolInfo):
    """Process data for a worker process."""

    MolIndex, SMILES = MolInfo

    MinHashFingerprint = GenerateMinHashFingerprintForMolecule(OptionsInfo["MinHashFingerprintsEncoder"], SMILES)
    if MinHashFingerprint is not None:
        MinHashFingerprint = MinHashFingerprint.tolist()

    return (MolIndex, MinHashFingerprint)


def SetupSMILESWithMolIndices(SMILES):
    """Setup an iterator to generate SMILES string along with a molecule index."""

    for MolIndex, MolSMILES in enumerate(SMILES):
        yield (MolIndex, MolSMILES)


def GenerateMinHashFingerprintForMolecule(MinHashFingerprintsEncoder, SMILES):
    """Generate MinHash fingerprint for a molecule."""

    MinHashFingerprint = None
    try:
        MinHashFingerprint = MinHashFingerprintsEncoder.encode(
            SMILES,
            radius=OptionsInfo["MinHashFPParams"]["Radius"],
            rings=OptionsInfo["MinHashFPParams"]["Rings"],
            kekulize=OptionsInfo["MinHashFPParams"]["Kekulize"],
            min_radius=OptionsInfo["MinHashFPParams"]["MinRadius"],
            sanitize=OptionsInfo["MinHashFPParams"]["Sanitize"],
        )
    except Exception as ErrMsg:
        if not OptionsInfo["QuietMode"]:
            MiscUtil.PrintWarning("Failed to generate MinHash fingerprint for SMILES %s:\n%s\n" % (SMILES, ErrMsg))
        else:
            MiscUtil.PrintInfo("")
        MinHashFingerprint = None

    return MinHashFingerprint


def RemoveFingerprintsFailedRows(InfileDF, FingerprintsFailedRowIndices):
    """Remove fingerprints failed rows."""

    if len(FingerprintsFailedRowIndices):
        InfileDF.drop(FingerprintsFailedRowIndices, inplace=True)
        InfileDF.reset_index(drop=True, inplace=True)


def GenerateTMAPCoordinates(LSHForest):
    """Generate TMAP coordinates."""

    MiscUtil.PrintInfo("\nGenerating TMAP plot coordinates...")

    PlotCoordsInfo = {}
    PlotCoordsInfo["NodeXCoords"] = None
    PlotCoordsInfo["NodeYCoords"] = None
    PlotCoordsInfo["EdgeNodeStartList"] = None
    PlotCoordsInfo["EdgeNodeToList"] = None

    LSHLayoutConfigParams = OptionsInfo["LSHLayoutConfigParams"]
    LSHLayoutConfig = tm.LayoutConfiguration()

    LSHLayoutConfig.k = LSHLayoutConfigParams["K"]
    LSHLayoutConfig.kc = LSHLayoutConfigParams["KC"]
    LSHLayoutConfig.fme_iterations = LSHLayoutConfigParams["FMEIterations"]
    LSHLayoutConfig.fme_randomize = LSHLayoutConfigParams["FMERandomize"]
    LSHLayoutConfig.fme_threads = LSHLayoutConfigParams["FMEThreads"]
    LSHLayoutConfig.fme_precision = LSHLayoutConfigParams["FMEPrecision"]
    LSHLayoutConfig.sl_repeats = LSHLayoutConfigParams["SLRepeats"]
    LSHLayoutConfig.sl_extra_scaling_steps = LSHLayoutConfigParams["SLExtraScalingSteps"]
    LSHLayoutConfig.sl_scaling_min = LSHLayoutConfigParams["SLScalingMin"]
    LSHLayoutConfig.sl_scaling_max = LSHLayoutConfigParams["SLScalingMax"]
    LSHLayoutConfig.sl_scaling_type = LSHLayoutConfigParams["SLScalingType"]
    LSHLayoutConfig.mmm_repeats = LSHLayoutConfigParams["MMMRepeats"]
    LSHLayoutConfig.placer = LSHLayoutConfigParams["Placer"]
    LSHLayoutConfig.merger = LSHLayoutConfigParams["Merger"]
    LSHLayoutConfig.merger_factor = LSHLayoutConfigParams["MergerFactor"]
    LSHLayoutConfig.merger_adjustment = LSHLayoutConfigParams["MergerAdjustment"]
    LSHLayoutConfig.node_size = 1.0 / LSHLayoutConfigParams["NodeSizeDenominator"]

    NodeXCoords, NodeYCoords, EdgeNodeStartList, EdgeNodeToList, _ = tm.layout_from_lsh_forest(
        LSHForest, config=LSHLayoutConfig
    )

    PlotCoordsInfo["NodeXCoords"] = NodeXCoords
    PlotCoordsInfo["NodeYCoords"] = NodeYCoords
    PlotCoordsInfo["EdgeNodeStartList"] = EdgeNodeStartList
    PlotCoordsInfo["EdgeNodeToList"] = EdgeNodeToList

    return PlotCoordsInfo


def SetupTMAPPlotData(InfileDF):
    """Setup plot data for TMAP plot."""

    MiscUtil.PrintInfo("\nSetting up TMAP plot data...")

    PlotDataInfo = {}
    PlotDataInfo["Columns"] = []
    PlotDataInfo["Colormaps"] = []
    PlotDataInfo["CategoricalStatus"] = []
    PlotDataInfo["LegendLabels"] = []
    PlotDataInfo["SeriesTitles"] = []

    # Setup categorical data...
    if OptionsInfo["CategoricalDataColnames"] is not None:
        for ColnameIndex, Colname in enumerate(OptionsInfo["CategoricalDataColnames"]):
            CategoryLabels, CategoryData = Faerun.create_categories(InfileDF[Colname])
            if len(CategoryLabels) > OptionsInfo["CategoricalDataMaxDisplay"]:
                CategoryLabels, CategoryData = RemapCategoricalPlotData(CategoryLabels, CategoryData)

            PlotDataInfo["Columns"].append(CategoryData)
            PlotDataInfo["Colormaps"].append(OptionsInfo["CategoricalDataColormapsList"][ColnameIndex])
            PlotDataInfo["CategoricalStatus"].append(True)
            PlotDataInfo["LegendLabels"].append(CategoryLabels)
            PlotDataInfo["SeriesTitles"].append(Colname)

    # Setup numerical data...
    if OptionsInfo["NumericalDataColnames"] is not None:
        for ColnameIndex, Colname in enumerate(OptionsInfo["NumericalDataColnames"]):
            PlotDataInfo["Columns"].append(InfileDF[Colname])
            PlotDataInfo["Colormaps"].append(OptionsInfo["NumericalDataColormapsList"][ColnameIndex])
            PlotDataInfo["CategoricalStatus"].append(False)
            PlotDataInfo["LegendLabels"].append(None)
            PlotDataInfo["SeriesTitles"].append(Colname)

    # Setup structure display data...
    FirstCol = True
    SMILESSelectedData = []
    SMILESSelectedLabels = []
    FirstCol = True
    for Colname in OptionsInfo["StructureDisplayDataColnames"]:
        if FirstCol:
            FirstCol = False
            SMILESSelectedData = InfileDF[Colname]
            SMILESSelectedLabels.append(Colname)
        else:
            SMILESSelectedData = SMILESSelectedData + "__" + InfileDF[Colname].astype(str)
            SMILESSelectedLabels.append(Colname)

    PlotDataInfo["SMILESSelectedData"] = SMILESSelectedData
    PlotDataInfo["SMILESSelectedLabels"] = SMILESSelectedLabels

    return PlotDataInfo


def RemapCategoricalPlotData(CategoryLabels, CategoryData):
    """Ramap categorical plot data."""

    if len(CategoryLabels) <= OptionsInfo["CategoricalDataMaxDisplay"]:
        return (CategoryLabels, CategoryData)

    # Track categories to remap...
    CategoryLabelsNew = []
    CategoryValuesToRemap = []
    LastCategoryValue = 0

    for CategoryLabelIndex, CategoryLabel in enumerate(CategoryLabels):
        CategoryValue, CategroyName = CategoryLabel
        if CategoryLabelIndex < OptionsInfo["CategoricalDataMaxDisplay"]:
            CategoryLabelsNew.append((CategoryValue, CategroyName))
            LastCategoryValue = CategoryValue
        else:
            CategoryValuesToRemap.append(CategoryValue)

    # Set up other category...
    OtherCategoryValue = LastCategoryValue + 1
    OtherCategoryName = "Other"
    CategoryLabelsNew.append((OtherCategoryValue, OtherCategoryName))

    # Update category labels and data...
    CategoryLabels = CategoryLabelsNew
    for ValueIndex, Value in enumerate(CategoryData):
        if Value in CategoryValuesToRemap:
            CategoryData[ValueIndex] = OtherCategoryValue

    return (CategoryLabels, CategoryData)


def GenerateTMAPPlot(InfileDF, PlotCoordsInfo, PlotDataInfo):
    """Generate TMAP plot."""

    MiscUtil.PrintInfo("\nGenerating TMAP plot...")

    # Initialize Faerun plot...
    FaerunConfigParams = OptionsInfo["FaerunConfigParams"]
    ImpressMsg = OptionsInfo["TMAPDisplayMsg"]
    TMAPFaerunPlot = Faerun(
        clear_color=FaerunConfigParams["ClearColor"],
        view="front",
        coords=False,
        title="",
        x_title="",
        y_title="",
        show_legend=FaerunConfigParams["ShowLegend"],
        legend_title=FaerunConfigParams["LegendTitle"],
        legend_orientation=FaerunConfigParams["LegendOrientation"],
        legend_number_format=FaerunConfigParams["LegendNumberFormat"],
        scale=FaerunConfigParams["Scale"],
        alpha_blending=FaerunConfigParams["AlphaBlending"],
        anti_aliasing=FaerunConfigParams["AntiAliasing"],
        thumbnail_width=FaerunConfigParams["ThumbnailWidth"],
        thumbnail_fixed=FaerunConfigParams["ThumbnailFixed"],
        impress=ImpressMsg,
    )

    # Setup scatter plot...
    ScatterPlotName = "Data"
    ScatterTreePlotName = "%s_tree" % ScatterPlotName
    FaerunScatterPlotParams = OptionsInfo["FaerunScatterPlotParams"]
    TMAPFaerunPlot.add_scatter(
        ScatterPlotName,
        {
            "x": PlotCoordsInfo["NodeXCoords"],
            "y": PlotCoordsInfo["NodeYCoords"],
            "c": PlotDataInfo["Columns"],
            "labels": PlotDataInfo["SMILESSelectedData"],
        },
        colormap=PlotDataInfo["Colormaps"],
        shader=FaerunScatterPlotParams["Shader"],
        point_scale=FaerunScatterPlotParams["PointScale"],
        max_point_size=FaerunScatterPlotParams["MaxPointSize"],
        fog_intensity=FaerunScatterPlotParams["FogIntensity"],
        categorical=PlotDataInfo["CategoricalStatus"],
        interactive=FaerunScatterPlotParams["Interactive"],
        has_legend=True,
        legend_labels=PlotDataInfo["LegendLabels"],
        series_title=PlotDataInfo["SeriesTitles"],
        selected_labels=PlotDataInfo["SMILESSelectedLabels"],
    )

    # Add scatter plot to Faerun...
    TMAPFaerunPlot.add_tree(
        ScatterTreePlotName,
        {"from": PlotCoordsInfo["EdgeNodeStartList"], "to": PlotCoordsInfo["EdgeNodeToList"]},
        point_helper=ScatterPlotName,
    )

    # Write out TMAP plot HTML and JS files...
    MiscUtil.PrintInfo("Writing TMAP plot files %s and %s..." % (OptionsInfo["Outfile"], OptionsInfo["OutfileJS"]))
    TMAPFaerunPlot.plot(OptionsInfo["OutfilePrefix"], template="smiles")

    if OptionsInfo["MergeHTMLandJSFilesMode"]:
        MergeTMAPResultsHTMLAndJSFiles()


def MergeTMAPResultsHTMLAndJSFiles():
    """Merge TMAP HTML and JS files."""

    MiscUtil.PrintInfo("\nMerging TMAP plot file %s into  %s..." % (OptionsInfo["OutfileJS"], OptionsInfo["Outfile"]))

    TMAPResultsHTMLFile = OptionsInfo["Outfile"]
    TMAPResultsJSFile = OptionsInfo["OutfileJS"]

    TMAPResultsTMPHTMLFile = "Tmp%s.html" % OptionsInfo["OutfilePrefix"]

    HTMLResultsFH = open(TMAPResultsHTMLFile, "r")
    JSResultsFH = open(TMAPResultsJSFile, "r")

    TMPHTMLResultsFH = open(TMAPResultsTMPHTMLFile, "w")

    for HTMLLine in HTMLResultsFH:
        HTMLLine = HTMLLine.rstrip()
        if re.search("%s" % TMAPResultsJSFile, HTMLLine, re.IGNORECASE):
            TMPHTMLResultsFH.write("    <script>\n")

            FirstLine = True
            for JSLine in JSResultsFH:
                JSLine = JSLine.rstrip()
                if FirstLine:
                    FirstLine = False
                    TMPHTMLResultsFH.write("    %s\n" % JSLine)
                else:
                    TMPHTMLResultsFH.write("%s\n" % JSLine)
            TMPHTMLResultsFH.write("\n    </script>\n")

        else:
            TMPHTMLResultsFH.write("%s\n" % HTMLLine)

    HTMLResultsFH.close()
    JSResultsFH.close()
    TMPHTMLResultsFH.close()

    MiscUtil.PrintInfo("Moving %s to %s..." % (TMAPResultsTMPHTMLFile, OptionsInfo["Outfile"]))
    shutil.move(TMAPResultsTMPHTMLFile, TMAPResultsHTMLFile)

    MiscUtil.PrintInfo("Removing %s file..." % (OptionsInfo["OutfileJS"]))
    os.remove(TMAPResultsJSFile)


def InitializeLSHForest():
    """Initialize LSH forest."""

    LSHForestParams = OptionsInfo["LSHForestParams"]
    LSHForest = tm.LSHForest(LSHForestParams["Dim"], LSHForestParams["NumPrefixTrees"], LSHForestParams["Store"])

    return LSHForest


def InitializeMinHashFingerprintsEncoder():
    """Initialize MinHash fingerprints encoder."""

    MinHashFPParams = OptionsInfo["MinHashFPParams"]
    MinHashFingerprintsEncoder = MHFPEncoder(
        n_permutations=MinHashFPParams["NumPermutations"], seed=MinHashFPParams["Seed"]
    )

    return MinHashFingerprintsEncoder


def ReadMoleculeData():
    """Read molecule data."""

    Infile = OptionsInfo["Infile"]
    InfileDelimiter = OptionsInfo["InfileDelimiter"]

    MiscUtil.PrintInfo("\nProcessing file %s..." % Infile)
    InfileDF = pd.read_csv(Infile, sep=InfileDelimiter)

    return InfileDF


def ProcessMolCountBasedAutoOptions(MolCount):
    """Process auto option values dependent on number of molecules."""

    #  Process "auto" option for LSHForestParams...
    ParamName = "NumPrefixTrees"
    ParamValue = "%s" % OptionsInfo["LSHForestParams"][ParamName]
    if re.match("^auto$", ParamValue, re.I):
        ParamValue = 128 if MolCount <= 10e03 else 8
        OptionsInfo["LSHForestParams"][ParamName] = ParamValue

    #  Process "auto" option for FaerunScatterPlotParams...
    ParamName = "PointScale"
    ParamValue = OptionsInfo["FaerunScatterPlotParams"][ParamName]
    ParamValue = "%s" % ParamValue
    if re.match("^auto$", ParamValue, re.I):
        if MolCount <= 10e03:
            ParamValue = 4.0
        elif MolCount <= 10e04:
            ParamValue = 2.0
        else:
            ParamValue = 1.0
        OptionsInfo["FaerunScatterPlotParams"][ParamName] = ParamValue

    #  Process "auto" option for LSHLayoutConfigParams...
    for ParamName in ["K", "KC", "SLRepeats", "SLExtraScalingSteps", "MMMRepeats", "NodeSizeDenominator"]:
        ParamValue = "%s" % OptionsInfo["LSHLayoutConfigParams"][ParamName]

        if not re.match("^auto$", ParamValue, re.I):
            continue

        if re.match("^K$", ParamName, re.I):
            ParamValue = 75 if MolCount <= 10e03 else 10
        elif re.match("^KC$", ParamName, re.I):
            ParamValue = 20 if MolCount <= 10e03 else 10
        elif re.match("^SLRepeats$", ParamName, re.I):
            ParamValue = 2 if MolCount <= 10e03 else 1
        elif re.match("^SLExtraScalingSteps$", ParamName, re.I):
            ParamValue = 4 if MolCount <= 10e03 else 2
        elif re.match("^MMMRepeats$", ParamName, re.I):
            ParamValue = 2 if MolCount <= 10e03 else 1
        elif re.match("^NodeSizeDenominator$", ParamName, re.I):
            ParamValue = 65.0 if MolCount <= 10e03 else 70.0

        OptionsInfo["LSHLayoutConfigParams"][ParamName] = ParamValue


def SetupTMAPDisplayMessage(MolCount, ValidMolCount):
    """Setup TMAP display message."""

    # Setup default TMAP display message using valid molecule count...
    if re.match("^auto$", OptionsInfo["TMAPDisplayMsg"], re.I):
        if MolCount == ValidMolCount:
            OptionsInfo["TMAPDisplayMsg"] = (
                "TMAP chemspace visualization<br/>Input file: %s<br/>Number of molecules: %s"
                % (OptionsInfo["Infile"], MolCount)
            )
        else:
            OptionsInfo["TMAPDisplayMsg"] = (
                "TMAP chemspace visualization<br/>Input file: %s<br/>Number of molecules: %s<br/>Number of valid molecules: %s"
                % (OptionsInfo["Infile"], MolCount, ValidMolCount)
            )


def ProcessFaerunConfigParametersOption():
    """Process option for faerun configuration parameters."""

    ParamsOptionName = "--faerunConfigParams"
    ParamsOptionValue = Options[ParamsOptionName]
    ParamsDefaultInfo = {
        "ClearColor": ["str", "#000000"],
        "ShowLegend": ["bool", True],
        "LegendTitle": ["str", "Legend"],
        "LegendOrientation": ["str", "vertical"],
        "LegendNumberFormat": ["str", "{:.2f}"],
        "Scale": ["float", 750.0],
        "AlphaBlending": ["bool", False],
        "AntiAliasing": ["bool", True],
        "ThumbnailWidth": ["int", 250],
        "ThumbnailFixed": ["bool", False],
    }

    FaerunConfigParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    ParamName = "LegendOrientation"
    ParamValue = FaerunConfigParams[ParamName]
    if not re.match("^(vertical|horizontal)$", ParamValue, re.I):
        MiscUtil.PrintError(
            'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: vertical or horizontal\n'
            % (ParamValue, ParamName, ParamsOptionName)
        )
    FaerunConfigParams[ParamName] = ParamValue.lower()

    for ParamName in ["Scale", "ThumbnailWidth"]:
        ParamValue = FaerunConfigParams[ParamName]
        if ParamValue <= 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )

    OptionsInfo["FaerunConfigParams"] = FaerunConfigParams


def ProcessFaerunScatterPlotParamsOption():
    """Process option for faerun scatter plot parameters."""

    ParamsOptionName = "--faerunScatterPlotParams"
    ParamsOptionValue = Options[ParamsOptionName]
    ParamsDefaultInfo = {
        "Shader": ["str", "circle"],
        "PointScale": ["str", "auto"],
        "MaxPointSize": ["float", 100.0],
        "FogIntensity": ["float", 0.0],
        "Interactive": ["bool", True],
    }

    FaerunScatterPlotParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    ParamName = "PointScale"
    ParamValue = FaerunScatterPlotParams[ParamName]
    if not re.match("^auto$", ParamValue, re.I):
        if not MiscUtil.IsFloat(ParamValue):
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option must be a float.'
                % (ParamValue, ParamName, ParamsOptionName)
            )
        ParamValue = float(ParamValue)
        if ParamValue <= 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )
        FaerunScatterPlotParams[ParamName] = ParamValue

    ParamName = "MaxPointSize"
    ParamValue = FaerunScatterPlotParams[ParamName]
    if ParamValue <= 0:
        MiscUtil.PrintError(
            'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
            % (ParamValue, ParamName, ParamsOptionName)
        )

    ParamName = "FogIntensity"
    ParamValue = FaerunScatterPlotParams[ParamName]
    if ParamValue < 0:
        MiscUtil.PrintError(
            'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: >= 0\n'
            % (ParamValue, ParamName, ParamsOptionName)
        )

    OptionsInfo["FaerunScatterPlotParams"] = FaerunScatterPlotParams


def ProcessLSHForestParamsOption():
    """Process option for LSH forest parameters."""

    ParamsOptionName = "--lshForestParams"
    ParamsOptionValue = Options[ParamsOptionName]
    ParamsDefaultInfo = {"Dim": ["int", 2048], "NumPrefixTrees": ["str", "auto"], "Store": ["bool", True]}

    LSHForestParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    ParamName = "Dim"
    ParamValue = LSHForestParams[ParamName]
    if ParamValue <= 0:
        MiscUtil.PrintError(
            'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
            % (ParamValue, ParamName, ParamsOptionName)
        )

    ParamName = "NumPrefixTrees"
    ParamValue = LSHForestParams[ParamName]
    if not re.match("^auto$", ParamValue, re.I):
        if not MiscUtil.IsInteger(ParamValue):
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option must be an integer.'
                % (ParamValue, ParamName, ParamsOptionName)
            )
        ParamValue = int(ParamValue)
        if ParamValue <= 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )
        LSHForestParams[ParamName] = ParamValue

    OptionsInfo["LSHForestParams"] = LSHForestParams


def ProcessLSHLayoutConfigParamsOption():
    """Process option for LSH configuration parameters."""

    ParamsOptionName = "--lshLayoutConfigParams"
    ParamsOptionValue = Options[ParamsOptionName]
    ParamsDefaultInfo = {
        "K": ["str", "auto"],
        "KC": ["str", "auto"],
        "FMEIterations": ["int", 1000],
        "FMERandomize": ["bool", False],
        "FMEThreads": ["int", 4],
        "FMEPrecision": ["int", 4],
        "SLRepeats": ["str", "auto"],
        "SLExtraScalingSteps": ["str", "auto"],
        "SLScalingMin": ["float", 1.0],
        "SLScalingMax": ["float", 1.0],
        "SLScalingType": ["str", "RelativeToDrawing"],
        "MMMRepeats": ["str", "auto"],
        "Placer": ["str", "Barycenter"],
        "Merger": ["str", "LocalBiconnected"],
        "MergerFactor": ["float", 2.0],
        "MergerAdjustment": ["int", 0],
        "NodeSizeDenominator": ["str", "auto"],
    }

    LSHLayoutConfigParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    for ParamName in [
        "FMEIterations",
        "FMEThreads",
        "FMEPrecision",
        "SLScalingMin",
        "SLScalingMax",
        "MergerFactor",
        "MergerAdjustment",
    ]:
        ParamValue = LSHLayoutConfigParams[ParamName]
        if re.match("^%s$" % ParamName, "MergerAdjustment", re.I):
            if ParamValue < 0:
                MiscUtil.PrintError(
                    'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: >= 0\n'
                    % (ParamValue, ParamName, ParamsOptionName)
                )
        else:
            if ParamValue <= 0:
                MiscUtil.PrintError(
                    'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                    % (ParamValue, ParamName, ParamsOptionName)
                )

    # Process "auto" values...
    for ParamName in ["K", "KC", "SLRepeats", "SLExtraScalingSteps", "MMMRepeats", "NodeSizeDenominator"]:
        ParamValue = LSHLayoutConfigParams[ParamName]

        if not re.match("^auto$", ParamValue, re.I):
            if re.match("^NodeSizeDenominator$", ParamName, re.I):
                if not MiscUtil.IsFloat(ParamValue):
                    MiscUtil.PrintError(
                        'The parameter value, %s, specified for parameter name, %s, using "%s" option must be a float.'
                        % (ParamValue, ParamName, ParamsOptionName)
                    )
                ParamValue = float(ParamValue)
            else:
                if not MiscUtil.IsInteger(ParamValue):
                    MiscUtil.PrintError(
                        'The parameter value, %s, specified for parameter name, %s, using "%s" option must be an integer.'
                        % (ParamValue, ParamName, ParamsOptionName)
                    )
                ParamValue = int(ParamValue)

            if ParamValue <= 0:
                MiscUtil.PrintError(
                    'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                    % (ParamValue, ParamName, ParamsOptionName)
                )
            LSHLayoutConfigParams[ParamName] = ParamValue

    # Map SLScalingType to TMAP object...
    ParamInfo = {
        "Absolute": tm.ScalingType.Absolute,
        "RelativeToAvgLength": tm.ScalingType.RelativeToAvgLength,
        "RelativeToDesiredLength": tm.ScalingType.RelativeToDesiredLength,
        "RelativeToDrawing": tm.ScalingType.RelativeToDrawing,
    }
    ParamName = "SLScalingType"
    MapLSHLayoutConfigParamToTMAPObject(LSHLayoutConfigParams, ParamsOptionName, ParamName, ParamInfo)

    # Map Placer to TMAP object...
    ParamInfo = {
        "Barycenter": tm.Placer.Barycenter,
        "Solar": tm.Placer.Solar,
        "Circle": tm.Placer.Circle,
        "Median": tm.Placer.Median,
        "Random": tm.Placer.Random,
        "Zero": tm.Placer.Zero,
    }
    ParamName = "Placer"
    MapLSHLayoutConfigParamToTMAPObject(LSHLayoutConfigParams, ParamsOptionName, ParamName, ParamInfo)

    # Map Merger to TMAP object...
    ParamInfo = {
        "EdgeCover": tm.Merger.EdgeCover,
        "LocalBiconnected": tm.Merger.LocalBiconnected,
        "Solar": tm.Merger.Solar,
        "IndependentSet": tm.Merger.IndependentSet,
    }
    ParamName = "Merger"
    MapLSHLayoutConfigParamToTMAPObject(LSHLayoutConfigParams, ParamsOptionName, ParamName, ParamInfo)

    OptionsInfo["LSHLayoutConfigParams"] = LSHLayoutConfigParams


def MapLSHLayoutConfigParamToTMAPObject(LSHLayoutConfigParams, ParamsOptionName, ParamName, ParamInfo):
    """Map LSH layout configuration patameter valut to TMAP object."""

    ParamValue = LSHLayoutConfigParams[ParamName]
    if ParamValue not in ParamInfo:
        MiscUtil.PrintError(
            'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: %s\n'
            % (ParamValue, ParamName, ParamsOptionName, ", ".join(sorted(ParamInfo.keys())))
        )
    LSHLayoutConfigParams[ParamName] = ParamInfo[ParamValue]


def ProcessMinHashFPParamsOption():
    """Process option for MinHash parameters."""

    ParamsOptionName = "--minHashFPParams"
    ParamsOptionValue = Options[ParamsOptionName]
    ParamsDefaultInfo = {
        "Radius": ["int", 3],
        "Rings": ["bool", True],
        "Kekulize": ["bool", True],
        "Sanitize": ["bool", True],
        "MinRadius": ["int", 1],
        "NumPermutations": ["int", 2048],
        "Seed": ["int", 42],
    }

    MinHashFPParams = MiscUtil.ProcessOptionNameValuePairParameters(
        ParamsOptionName, ParamsOptionValue, ParamsDefaultInfo
    )

    for ParamName in ["Radius", "MinRadius", "NumPermutations"]:
        ParamValue = MinHashFPParams[ParamName]
        if ParamValue <= 0:
            MiscUtil.PrintError(
                'The parameter value, %s, specified for parameter name, %s, using "%s" option is not a valid value. Supported values: > 0\n'
                % (ParamValue, ParamName, ParamsOptionName)
            )

    OptionsInfo["MinHashFPParams"] = MinHashFPParams


def ProcessInfileDelimiterOption():
    """Process option infile delimiter."""

    InfileDelim = Options["--infileDelimiter"]
    if re.match("^auto$", InfileDelim, re.I):
        FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["Infile"])
        if re.match("^csv$", FileExt, re.I):
            InfileDelim = "comma"
        elif re.match("^(tsv|txt)$", FileExt, re.I):
            InfileDelim = "tab"
        elif re.match("^(smi)$", FileExt, re.I):
            InfileDelim = "space"
        else:
            MiscUtil.PrintError(
                'The input file delimiter couldn\'t be determined from its extension %s. You must explicitly specify an input file delimiter using option"--infileDelimiter".\n'
                % (InfileDelim)
            )

    InfileDelimMap = {"comma": ",", "tab": "\t", "space": " "}
    OptionsInfo["InfileDelimiter"] = InfileDelimMap[InfileDelim]


def ProcessColumnModeOption():
    """Process column mode option."""

    CollabelMode, ColnumMode = [False, False]
    Colmode = Options["--colmode"]
    if re.match("^collabel$", Colmode, re.I):
        CollabelMode = True
    elif re.match("^colnum$", Colmode, re.I):
        ColnumMode = True
    else:
        MiscUtil.PrintError(
            'The value, %s, specified for option "-c, --colmode" is not valid. Supported values: collabel or colnum\n'
            % (Colmode)
        )

    OptionsInfo["Colmode"] = Colmode
    OptionsInfo["CollabelMode"] = CollabelMode
    OptionsInfo["ColnumMode"] = ColnumMode


def RetrieveColumnNames():
    """Retrieve column names."""

    Infile = OptionsInfo["Infile"]

    InfileFH = open(Infile, "r")
    InfileReader = csv.reader(InfileFH, delimiter=OptionsInfo["InfileDelimiter"], quotechar='"')
    Colnames = next(InfileReader)
    InfileFH.close()

    if len(Colnames) == 0:
        MiscUtil.PrintError("The first line in input file, %s, is empty. It must contain column names.\n" % Infile)

    ColnameToColnumMap = {}
    ColnumToColnameMap = {}
    for ColIndex, Colname in enumerate(Colnames):
        Colnum = ColIndex + 1
        ColnameToColnumMap[Colname] = Colnum
        ColnumToColnameMap[Colnum] = Colname

    OptionsInfo["Colnames"] = Colnames
    OptionsInfo["ColCount"] = len(Colnames)
    OptionsInfo["ColnameToColnumMap"] = ColnameToColnumMap
    OptionsInfo["ColnumToColnameMap"] = ColnumToColnameMap

    # Initialize for tracking specified column names...
    SpecifiedColsInfo = {}
    SpecifiedColsInfo["Colnames"] = []
    SpecifiedColsInfo["Colnum"] = {}
    SpecifiedColsInfo["OptionName"] = {}

    OptionsInfo["SpecifiedColsInfo"] = SpecifiedColsInfo


def ProcessSMILESColOption():
    """Process SMILES column option."""

    SMILESCol = Options["--colSMILES"]
    if re.match("^auto$", SMILESCol, re.I):
        Colname = "SMILES"
        if Colname not in OptionsInfo["ColnameToColnumMap"]:
            MiscUtil.PrintError(
                'The SMILES column name, %s, doen\'t exist in input file. You must specify a valid SMILES column name or number using "--colSMILES" option.\n'
                % Colname
            )

        Colnum = OptionsInfo["ColnameToColnumMap"][Colname]
        SMILESColspec = Colnum if OptionsInfo["ColnumMode"] else Colname
    else:
        SMILESColspec = SMILESCol

    SMILESColname, SMILESColnum = ProcessColumnSpecification("--colSMILES", SMILESColspec)

    OptionsInfo["SMILESCol"] = SMILESCol
    OptionsInfo["SMILESColname"] = SMILESColname
    OptionsInfo["SMILESColnum"] = SMILESColnum


def ProcessCategoricalDataColsOption():
    """Process categorical data columns option."""

    CategoricalDataColnames, CategoricalDataColnums = [None] * 2
    CategoricalDataCols = Options["--categoricalDataCols"]
    if not re.match("^none$", CategoricalDataCols, re.I):
        CategoricalDataColnames = []
        CategoricalDataColnums = []
        for DataCol in CategoricalDataCols.split(","):
            DataCol = DataCol.strip()
            DataColname, DataColnum = ProcessColumnSpecification("--categoricalDataCols", DataCol)
            CategoricalDataColnames.append(DataColname)
            CategoricalDataColnums.append(DataColnum)

    OptionsInfo["CategoricalDataCols"] = CategoricalDataCols
    OptionsInfo["CategoricalDataColnames"] = CategoricalDataColnames
    OptionsInfo["CategoricalDataColnums"] = CategoricalDataColnums


def ProcessCategoricalDataColormapsOption():
    """Process categorical data color maps option."""

    if OptionsInfo["CategoricalDataColnames"] is None:
        OptionsInfo["CategoricalDataColormaps"] = Options["--categoricalDataColormaps"]
        OptionsInfo["CategoricalDataColormapsList"] = None
        return

    CategoricalDataColormapsList = []
    CategoricalDataColCount = len(OptionsInfo["CategoricalDataColnames"])

    CategoricalDataColormaps = Options["--categoricalDataColormaps"]
    if not re.match("^auto$", CategoricalDataColormaps, re.I):
        ColormapsWords = CategoricalDataColormaps.split(",")
        if len(ColormapsWords) != CategoricalDataColCount:
            MiscUtil.PrintInfo(
                'The number of colormaps, %s, specified using "--categoricalDataColormaps" must be equal to the number of columns, %s, specified using "--categoricalDataCols" option.'
                % (len(ColormapsWords), CategoricalDataColCount)
            )
        for Colormap in ColormapsWords:
            Colormap = Colormap.strip()
            CategoricalDataColormapsList.append(Colormap)
    else:
        CategoricalDataColormapsList = ["tab10"] * CategoricalDataColCount

    OptionsInfo["CategoricalDataColormaps"] = CategoricalDataColormaps
    OptionsInfo["CategoricalDataColormapsList"] = CategoricalDataColormapsList


def ProcessNumericalDataColsOption():
    """Process numerical data columns option."""

    NumericalDataColnames, NumericalDataColnums = [None] * 2
    NumericalDataCols = Options["--numericalDataCols"]
    if not re.match("^none$", NumericalDataCols, re.I):
        NumericalDataColnames = []
        NumericalDataColnums = []
        for DataCol in NumericalDataCols.split(","):
            DataCol = DataCol.strip()
            DataColname, DataColnum = ProcessColumnSpecification("--numericalDataCols", DataCol)
            NumericalDataColnames.append(DataColname)
            NumericalDataColnums.append(DataColnum)

    OptionsInfo["NumericalDataCols"] = NumericalDataCols
    OptionsInfo["NumericalDataColnames"] = NumericalDataColnames
    OptionsInfo["NumericalDataColnums"] = NumericalDataColnums


def ProcessNumericalDataColormapsOption():
    """Process numerical data color maps option."""

    if OptionsInfo["NumericalDataColnames"] is None:
        OptionsInfo["NumericalDataColormaps"] = Options["--numericalDataColormaps"]
        OptionsInfo["NumericalDataColormapsList"] = None
        return

    NumericalDataColormapsList = []
    NumericalDataColCount = len(OptionsInfo["NumericalDataColnames"])

    NumericalDataColormaps = Options["--numericalDataColormaps"]
    if not re.match("^auto$", NumericalDataColormaps, re.I):
        ColormapsWords = NumericalDataColormaps.split(",")
        if len(ColormapsWords) != NumericalDataColCount:
            MiscUtil.PrintInfo(
                'The number of colormaps, %s, specified using "--categoricalDataColormaps" must be equal to the number of columns, %s, specified using "--categoricalDataCols" option.'
                % (len(ColormapsWords), NumericalDataColCount)
            )
        for Colormap in ColormapsWords:
            Colormap = Colormap.strip()
            NumericalDataColormapsList.append(Colormap)
    else:
        NumericalDataColormapsList = ["viridis"] * NumericalDataColCount

    OptionsInfo["NumericalDataColormaps"] = NumericalDataColormaps
    OptionsInfo["NumericalDataColormapsList"] = NumericalDataColormapsList


def ProcessStructureDisplayDataColsOption():
    """Process structure display data columns option."""

    StructureDisplayDataColnames = []
    StructureDisplayDataColnums = []

    # Add SMILES column...
    StructureDisplayDataColnames.append(OptionsInfo["SMILESColname"])
    StructureDisplayDataColnums.append(OptionsInfo["SMILESColnum"])

    # Process specified columns...
    OptionName = "--structureDisplayDataCols"
    StructureDisplayDataCols = Options[OptionName]
    if re.match("^auto$", StructureDisplayDataCols, re.I):
        # Automatically add 'Name' column...
        Colname = "Name"
        if Colname in OptionsInfo["ColnameToColnumMap"]:
            Colnum = OptionsInfo["ColnameToColnumMap"][Colname]
            StructureDisplayDataColnames.append(Colname)
            StructureDisplayDataColnums.append(Colnum)
    else:
        for DataCol in StructureDisplayDataCols.split(","):
            DataCol = DataCol.strip()
            if OptionsInfo["ColnumMode"]:
                Colnum = int(DataCol)
                if Colnum not in OptionsInfo["ColnumToColnameMap"]:
                    MiscUtil.PrintError(
                        'The column number, %s, specified using "%s" option doesn\'t exist in input file. You must specify a valid column number. Valid values: >= 1 and <= %s\n'
                        % (Colnum, OptionName, OptionsInfo["ColCount"])
                    )
                Colname = OptionsInfo["ColnumToColnameMap"][Colnum]
            else:
                Colname = DataCol
                if Colname not in OptionsInfo["ColnameToColnumMap"]:
                    MiscUtil.PrintError(
                        'The column name, %s, specified using "%s" option doesn\'t exist in input file. You must specify a valid column name. Valid values: %s\n'
                        % (Colname, OptionName, " ".join(OptionsInfo["Colnames"]))
                    )
                Colnum = OptionsInfo["ColnameToColnumMap"][Colname]

            if Colname in StructureDisplayDataColnames:
                StructureDisplayDataColnumsStrs = ["%s" % Num for Num in StructureDisplayDataColnums]
                if OptionsInfo["ColnumMode"]:
                    MiscUtil.PrintError(
                        'The column number, %s, specified using "%s" option is a duplicate column number. It has already been used for this option. You must specify a different column number. Used column names: %s; Used column nums: %s\n'
                        % (
                            Colnum,
                            OptionName,
                            " ".join(StructureDisplayDataColnames),
                            " ".join(StructureDisplayDataColnumsStrs),
                        )
                    )
                else:
                    MiscUtil.PrintError(
                        'The column name, %s, specified using "%s" option is a duplicate column name. It has already been used for this option. You must specify a different column name. Used column names: %s; Used column nums: %s\n'
                        % (
                            Colname,
                            OptionName,
                            " ".join(StructureDisplayDataColnames),
                            " ".join(StructureDisplayDataColnumsStrs),
                        )
                    )

            StructureDisplayDataColnames.append(Colname)
            StructureDisplayDataColnums.append(Colnum)

    OptionsInfo["StructureDisplayDataCols"] = StructureDisplayDataCols
    OptionsInfo["StructureDisplayDataColnames"] = StructureDisplayDataColnames
    OptionsInfo["StructureDisplayDataColnums"] = StructureDisplayDataColnums


def ProcessColumnSpecification(OptionName, Colspec):
    """Process column specification corresponding to a column name or number."""

    Colname, Colnum = [None, None]
    if OptionsInfo["ColnumMode"]:
        Colnum = int(Colspec)
        if Colnum not in OptionsInfo["ColnumToColnameMap"]:
            MiscUtil.PrintError(
                'The column number, %s, specified using "%s" option doesn\'t exist in input file. You must specify a valid column number. Valid values: >= 1 and <= %s\n'
                % (Colnum, OptionName, OptionsInfo["ColCount"])
            )
        Colname = OptionsInfo["ColnumToColnameMap"][Colnum]
    else:
        Colname = Colspec
        if Colname not in OptionsInfo["ColnameToColnumMap"]:
            MiscUtil.PrintError(
                'The column name, %s, specified using "%s" option doesn\'t exist in input file. You must specify a valid column name. Valid values: %s\n'
                % (Colname, OptionName, " ".join(OptionsInfo["Colnames"]))
            )
        Colnum = OptionsInfo["ColnameToColnumMap"][Colname]

    # Track and check for duplicate column specification...
    SpecifiedColsInfo = OptionsInfo["SpecifiedColsInfo"]
    if Colname in SpecifiedColsInfo["Colnames"]:
        if OptionsInfo["ColnumMode"]:
            MiscUtil.PrintError(
                'The column number, %s, specified using "%s" option is a duplicate column number. It has already been used for "%s" option. You must specify a different column number.\n'
                % (Colnum, OptionName, SpecifiedColsInfo["OptionName"][Colname])
            )
        else:
            MiscUtil.PrintError(
                'The column name, %s, specified using "%s" option is a duplicate column name. It has already been used for "%s" option. You must specify a different column name.\n'
                % (Colname, OptionName, SpecifiedColsInfo["OptionName"][Colname])
            )
    else:
        SpecifiedColsInfo["Colnames"].append(Colname)
        SpecifiedColsInfo["Colnum"][Colname] = Colnum
        SpecifiedColsInfo["OptionName"][Colname] = OptionName

    return (Colname, Colnum)


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    # Validate options...
    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]

    Outfile = Options["--outfile"]
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--outfile"])
    OptionsInfo["OutfilePrefix"] = FileName
    OptionsInfo["OutfileExt"] = FileExt

    OptionsInfo["Outfile"] = Outfile
    OptionsInfo["OutfileJS"] = "%s.js" % FileName
    OptionsInfo["OutfileLSHForest"] = "%s.dat" % FileName

    ProcessInfileDelimiterOption()
    RetrieveColumnNames()

    ProcessColumnModeOption()
    ProcessSMILESColOption()

    OptionsInfo["CategoricalDataMaxDisplay"] = int(Options["--categoricalDataMaxDisplay"])
    ProcessCategoricalDataColsOption()
    ProcessCategoricalDataColormapsOption()

    ProcessNumericalDataColsOption()
    ProcessNumericalDataColormapsOption()

    ProcessStructureDisplayDataColsOption()

    ProcessFaerunConfigParametersOption()
    ProcessFaerunScatterPlotParamsOption()

    OptionsInfo["LSHForestFileWriteMode"] = True if re.match("^yes$", Options["--lshForestFileWrite"], re.I) else False
    OptionsInfo["LSHForestFileRestoreMode"] = (
        True if re.match("^yes$", Options["--lshForestFileRestore"], re.I) else False
    )
    if OptionsInfo["LSHForestFileRestoreMode"]:
        LSHForestFile = OptionsInfo["OutfileLSHForest"]
        if not os.path.isfile(LSHForestFile):
            MiscUtil.PrintError(
                'The LSH forest file, %s, must be present for, %s, value of "--lshForestFileRestore" option.'
                % (LSHForestFile, Options["--lshForestFileRestore"])
            )

    ProcessLSHForestParamsOption()
    ProcessLSHLayoutConfigParamsOption()

    OptionsInfo["MergeHTMLandJSFilesMode"] = (
        True if re.match("^yes$", Options["--mergeHTMLandJSFiles"], re.I) else False
    )

    ProcessMinHashFPParamsOption()

    OptionsInfo["MPMode"] = True if re.match("^yes$", Options["--mp"], re.I) else False
    OptionsInfo["MPParams"] = MiscUtil.ProcessOptionMultiprocessingParameters("--mpParams", Options["--mpParams"])

    OptionsInfo["Overwrite"] = Options["--overwrite"]
    OptionsInfo["QuietMode"] = True if re.match("^yes$", Options["--quiet"], re.I) else False

    OptionsInfo["TMAPDisplayMsg"] = Options["--tmapDisplayMsg"]


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
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "smi csv tsv txt")

    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "html")
    MiscUtil.ValidateOptionsOutputFileOverwrite(
        "-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"]
    )
    MiscUtil.ValidateOptionsDistinctFileNames(
        "-i, --infile", Options["--infile"], "-o, --outfile", Options["--outfile"]
    )

    MiscUtil.ValidateOptionTextValue("-c, --colmode", Options["--colmode"], "collabel colnum")

    if re.match("^none$", Options["--categoricalDataCols"], re.I) and re.match(
        "^none$", Options["--numericalDataCols"], re.I
    ):
        MiscUtil.PrintError(
            'You must specify al least one caetgorical or numerical data column using option "--categoricalDataCols" or "--numericalDataCols". It is used to color TMAP.'
        )

    ColnumMode = True if re.match("^colnum$", Options["--colmode"], re.I) else False
    if ColnumMode and not re.match("^auto$", Options["--colSMILES"], re.I):
        MiscUtil.ValidateOptionIntegerValue("--colSMILES", Options["--colSMILES"], {">": 0})

    if ColnumMode and not re.match("^none$", Options["--categoricalDataCols"], re.I):
        MiscUtil.ValidateOptionNumberValues(
            "--categoricalDataCols", Options["--categoricalDataCols"], 0, ",", "integer", {">": 0}
        )

    MiscUtil.ValidateOptionIntegerValue("--categoricalDataMaxDisplay", Options["--categoricalDataMaxDisplay"], {">": 0})

    if not re.match("^auto$", Options["--categoricalDataColormaps"], re.I):
        ColormapCount = len(Options["--categoricalDataColormaps"].split(","))
        ColCount = len(Options["--categoricalDataCols"].split(","))
        if ColormapCount != ColCount:
            MiscUtil.PrintError(
                'The number of colormaps, %s, specified using option "--categoricalDataColormaps" must be equal to number of columns, %s,  specified using option "-categoricalDataCols". '
                % (ColormapCount, ColCount)
            )

    if ColnumMode and not re.match("^none$", Options["--numericalDataCols"], re.I):
        MiscUtil.ValidateOptionNumberValues(
            "--numericalDataCols", Options["--numericalDataCols"], 0, ",", "integer", {">": 0}
        )

    if not re.match("^auto$", Options["--numericalDataColormaps"], re.I):
        ColormapCount = len(Options["--numericalDataColormaps"].split(","))
        ColCount = len(Options["--numericalDataCols"].split(","))
        if ColormapCount != ColCount:
            MiscUtil.PrintError(
                'The number of colormaps, %s, specified using option "--numericalDataColormaps" must be equal to number of columns, %s,  specified using option "-numericalDataCols". '
                % (ColormapCount, ColCount)
            )

    if not re.match("^auto$", Options["--structureDisplayDataCols"], re.I):
        if ColnumMode and not re.match("^none$", Options["--structureDisplayDataCols"], re.I):
            MiscUtil.ValidateOptionNumberValues(
                "--structureDisplayDataCols", Options["--structureDisplayDataCols"], 0, ",", "integer", {">": 0}
            )

    if not re.match("^auto$", Options["--infileDelimiter"], re.I):
        MiscUtil.ValidateOptionTextValue(" --infileDelimiter", Options["--infileDelimiter"], "comma tab space")

    MiscUtil.ValidateOptionTextValue("--lshForestFileWrite", Options["--lshForestFileWrite"], "yes no")
    MiscUtil.ValidateOptionTextValue("--lshForestFileRestore", Options["--lshForestFileRestore"], "yes no")
    MiscUtil.ValidateOptionTextValue("--mergeHTMLandJSFiles", Options["--mergeHTMLandJSFiles"], "yes no")

    MiscUtil.ValidateOptionTextValue("--mp", Options["--mp"], "yes no")


# Setup a usage string for docopt...
_docoptUsage_ = """
VisualizeChemspaceUsingTMAP.py - Visualize chemspace

Usage:
    VisualizeChemspaceUsingTMAP.py [--categoricalDataCols <collabel1,... or colnum1,...>] [--categoricalDataColormaps <Colormap1, Colormap2,...>]
                                   [--categoricalDataMaxDisplay <number>] [--colmode <collabel or colnum>] [--colSMILES <text or number>]
                                   [--faerunConfigParams <Name,Value,...>] [--faerunScatterPlotParams <Name,Value,...>]
                                   [--infileDelimiter <comma, tab, or space>] [--lshForestFileWrite <yes or no>] [--lshForestFileRestore <yes or no>]
                                   [--lshForestParams <Name,Value,...>] [--lshLayoutConfigParams  <Name,Value,...>] [--mergeHTMLandJSFiles <yes or no>]
                                   [--minHashFPParams <Name,Value,...>] [--mp <yes or no>] [--mpParams <Name,Value,...>]
                                   [--numericalDataCols <collabel1,... or colnum1,...>] [--numericalDataColormaps <Colormap1, Colormap2,...>]
                                   [--overwrite] [--quiet <yes or no>] [--structureDisplayDataCols <collabel1,... or colnum1,...> ]
                                   [--tmapDisplayMsg <text>] [-w <dir>] -i <infile> -o <outfile> 
    VisualizeChemspaceUsingTMAP.py -h | --help | -e | --examples

Description:
    Generate an interactive TreeMAP (TMAP) [Ref 171, 172] visualization for molecules
    in a text input file. The text input file must have a column containing SMILES strings.
    In addition, it must contain at least one column corresponding to categorical or
    numerical data for coloring TMAP nodes. You may optionally map multiple categorical
    and numerical data columns on to a TMAP visualization. A HTML file is generated for
    interactive visualization of chemspace in a browser.

    The TMAP methodology is able to generate a reasonably interactive visualization
    for relatively large data sets. A brief description of the methodology is as follows.
    A set of MinHash Fingerprints (MHFPs) are calculated for molecules in input file
    followed by the generation of a Locality Sensitivity Hashing (LSH) forest employing
    MHFPs. A c-approximate k-Nearest Neighbor Graph (c-k-NNG) is constructed from
    LSH, which is used to construct a Minimum Spanning Tree (MST) or Forest (MSF).
    The final TMAP visualization is generated by laying out MST and MSF on a plane
    using an algorithm provided by the Open Graph Drawing Framework (OGDF). The
    OGDF provides flexibility to adjust graph layout methodology in terms of not only
    aesthetics but also computational time.

    The supported input file formats are: CSV (.csv) TSV (.txt or .tsv),
    SMILES (.smi)

    The supported output file format is: HTML (.html).

Options:
    --categoricalDataCols <collabel1,... or colnum1,...>  [default: none]
        A comma delimited list of column labels or numbers corresponding to
        categorical data to map on a TMAP visualization.
    --categoricalDataColormaps <Colormap1, Colormap2,...>  [default: auto]
        A comma delimited list of color map names corresponding to categorical
        data. The default is to use 'tab10' color map name for mapping categorical
        data on a TMAP. The number of specified color maps must match the number
        of categorical data columns. You must specify valid color map names
        supported by Matplotlib. No validation is performed. Example color map
        names for categorical data: Pastel1, Pastel2, Paired, Accent, Dark2, Set1,
        Set2, Set3, tab10, tab20, tab20b, tab20c.
    --categoricalDataMaxDisplay <number>  [default: 6]
        Maximum number of categories in a category column to display on a TMAP
        visualization. The rest of the categories are aggregated under a new
        category named 'Other' before mapping on to a TMAP visualization.
    -c, --colmode <collabel or colnum>  [default: collabel]
        Use column number or name for the specification of columns in input
        text file containing SMILES strings and molecule names along with any 
        categorical or numerical data.
    --colSMILES <text or number>  [default: auto]
        Column name or number corresponding to SMILES strings. The default value
        is automatically set based on the value of '-c, --colmode': 'SMILES'  for
        'collabel'; SMILES string column number for 'colnum'. SMILES strings must
        be present in input file.
    -e, --examples
        Print examples.
    --faerunConfigParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for configuring
        faerun (Ref 172) to generate a TMAP visualization.
        
        The supported parameter names along with their default and possible
        values are shown below:
             
            clearColor, #000000
            showLegend, yes  [ Possible values: yes or no ] 
            legendTitle, Legend
            legendOrientation, vertical  [ Possible values: vertical or
                horizontal ]
            legendNumberFormat, {:.2f}
            scale, 750.0
            alphaBlending, no  [ Possible values: yes or no ]
            antiAliasing, yes  [Possible values: yes or no]
            thumbnailWidth, 250
            thumbnailFixed, no  [ Possible values: yes or no ]
            
        A brief description of parameters, as available in the code for faerun, is
        provided below:
        
            clearColor: Background color
            showLegend: Show legend at lower right
            legendTitle: Legend title
            legendOrientation: Legend Orientation
            legendNumberFormat: Number string format applied to numbers
                displayed in legend
            scale: Scaling factor for scaling normalized coordinates
            AlphaBlending: Activate alpha blending. It is required for smoothCircle
                shader.
            antiAliasing: Activate anti-aliasing. It might adversly impact
                rendering performance.
            thumbnailWidth: Width of thumbnail images for structures
            thumbnailFixed:  Show thumbnail images at a fixed location at the
                top instead of next to the mouse
            
    --faerunScatterPlotParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        scatter plot representing a TMAP using faerun (Ref 172).
        
        The supported parameter names along with their default and possible
        values are shown below:
             
            shader, circle  [ Possible values: circle, smoothCircle,
                sphere, or any valid value]
            pointScale, auto  [ 4 if MolCout<=10K; 2 if MolCount<=100K; else 1 ]
            maxPointSize, 100.0
            fogIntensity, 0.0
            interactive, yes  [ Possible values: yes or no ] 
             
        A brief description of parameters is provided below:
        
            shader: Shader to use for visualizating data points
            pointScale: Relative size of data points
            maxPointSize: Maximum size of the data points during zooming
            fogIntensity: Intensity of distance fog
            interactive: Generate interactive scatter plot
            
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name. The SMILES strings must be present in the input file.
        Supported formats: CSV (.csv) TSV (.txt or .tsv), or SMILES (.smi)
    --infileDelimiter <comma, tab, or space>  [default: auto]
        Input file delimiter for processing data. The default value is automatically
        set based on the type of input file: comma - CSV (.csv); tab - TSV (.txt or
        .tsv);  space - SMILES (.smi)
    --lshForestFileWrite <yes or no>  [default: yes]
        Write LSH forest data a file for subsequent generation of a TMAP visualization.
        Default file name: <OutfileRoot>_LSHForest.dat. The LSH forest data is
        generated using MinHash fingerprints. You may restore LSH forest data
        using '--lshForestFileRestore' option to skip the generation of fingerprints.
    --lshForestFileRestore <yes or no>  [default: no]
        Check and restore LSH forest data from a file for generating a TMAP
        visualization and skip the generation of MinHash fingerprints. Default file
        name: <OutfileRoot>_LSHForest.dat
    --lshForestParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        LSH (Locality Sensitivity Hashing) forest from MinHash fingerprints.
        
        The supported parameter names along with their default and possible
        values are shown below:
             
            dim, 2048
            numPrefixTrees, auto  [ 128 if MolCount <= 10K else 8 ]
            store, yes  [ Possible values: yes or no ]
            
        A brief description of parameters, as available in the code for LSH, is
        provided below:
        
            dim: Dimensionality of MinHashes to be added to LSHForest
            numPrefixTrees: Number of prefix trees to use
            store: store the data for enhanced retrieval
            
    --lshLayoutConfigParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for configuring
        LSH (Locality Sensitivity Hashing) layout.
        
        The supported parameter names along with their default and possible
        values are shown below:
            
            k, auto  [ 75 if MolCount <= 10K else 10]
            kc, auto  [ 20 if MolCount <= 10K else 10]
            fmeIterations, 1000
            fmeRandomize, no  [ Possible values: yes or no ]
            fmeThreads, 4
            fmePrecision, 4
            slRrepeats, auto  [ 2 if MolCount <= 10K else 1]
            slExtraScalingSteps, auto  [ 4 if MolCount <= 10K else 2 ]
            slScalingMin, 1.0
            slScalingMax, 1.0
            slScalingType, RelativeToDrawing  [ Possible values: Absolute,
                RelativeToAvgLength, RelativeToDesiredLength, or
                RelativeToDrawing ]
            mmmRepeats, auto  [ 2 MolCount <= 10K else 1 ]
            placer, Barycenter  [ Possible valeues: Barycenter, Solar, Circle,
                Median, Random, or Zero ]
            merger, LocalBiconnected  [ Possible values: EdgeCover,
                LocalBiconnected, Solar, or IndependentSet ]
            mergerFactor, 2.0
            mergerAdjustment, 0
            nodeSizeDenominator, auto  [ 65 if MolCout <= 10K else 70.0]
            
        A brief description of parameters, as available in the code for LSH, is
        provided below:
            
            k: Number of nearest neighbors used to create k-nearest neighbor
                graph
            kc: Scalar by which k is multiplied before querying LSH forest.
                The results are then sorted in decreasing order based on linear
                scan distances. 
            fmeIterations: Maximum number of iterations of Fast Multipole
                Embedder (FME)
            fmeRandomize: Randomize FME layout at the start
            fmeThreads: Number of threads for FME
            fmePrecision: Number of coefficients of multipole expansion
            slRepeats: Number of repeats of scaling layout algorithm
            slExtraScalingSteps: Number of repeats of scaling
            slScalingMin: Minimum scaling factor
            slScalingMax: Maximum scaling factor.
            slScalingType: Scaling type corresponding to relative scale of graph
            mmmRepeats, Number of repeats of layout at each level
            placer: Methodology for defining initial positions of vertices in a
                graph at each level
            merger: Vertex merging methodology used during coarsening phase
                of multilevel algorithm
            mergerFactor: Ratio of sizes between two levels up to which merging
                is performed.  It doesn't apply to all merging methodologies.
            mergerAdjustment: Edge  length  adjustment  for merging methodology.
                It doesn't apply to all merging methodologies.
            nodeSizeDenominator: Node size denominator affecting the magnitude
                of repelling force between nodes. Node size corresponds to
                1.0 / nodeSizeDenominator. You may want to increase the value
                nodeSizeDenominator to decrease node size and resolve overlaps
                in  a crowded tree.
            
    --mergeHTMLandJSFiles <yes or no>  [default: yes]
        Merge TMAP JS data file into HTML file and delete JS data file. Default
        file names: <OutfileRoot>.html, <OutfileRoot>.js.
    --minHashFPParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        Min Hash Fingerprints (MHFP).
        
        The supported parameter names along with their default and possible
        values are shown below:
            
            radius, 3
            rings, yes  [ Possible values: yes or no ]
            kekulize, yes  [ Possible values: yes or no ]
            sanitize, yes  [ Possible values: yes or no ]
            minRadius, 1
            numPermutations, 2048
            seed, 42
            
        A brief description of parameters, as available in the code for MHFP,  is
        provided below:
            
            radius:  MHFP radius (A radius of 3 corresponds to MHFP6)
            rings:  Include rings in shingling
            kekulize:  Kekulize SMILES
            sanitize:  Sanitize SMILES
            minRadius: Minimum radius that is used to extract n-grams
            numPermutations: Number of permutations used for hashing
            seed: Random number seed for numpy.random
            
    --mp <yes or no>  [default: no]
        Use multiprocessing for the generation of fingerprints.
         
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
        multiprocessing during the generation of fingerprints.
        
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
    --numericalDataCols <collabel1,... or colnum1,...>  [default: none]
        A comma demlimited list of column labels or numbers corresponding to
        numerical data to map on a TMAP visualization.
    --numericalDataColormaps <Colormap1, Colormap2,...>  [default: auto]
        A comma demlimited list of color map names corresponding to numerical
        data. The default is to use 'viridis' color map name for mapping numerical
        data on a TMAP. The number of specified color maps must mtach the number
        of numerical data columns. You must specify valid color map names
        supported by Matplotlib. No validation is performed. Example color map
        names for numerical data: viridis, plasma, inferno, magma, cividis.
    -o, --outfile <outfile>
        Output HTML file name for writing out a TMAP visualization.
    --overwrite
        Overwrite existing files.
    -q, --quiet <yes or no>  [default: no]
        Use quiet mode. The warning and information messages will not be printed.
    --structureDisplayDataCols <collabel1,... or colnum1,...>  [default: auto]
        A comma delimited list of column labels or numbers corresponding to data
        to display under a thumbnail image of a structure in a TMAP visualization.
        The default column is set to 'Name' and it is automatically shown. In addition,
        the SMILES string column is always used to display SMILES under the structures.
    -t, --tmapDisplayMsg <text>  [default: auto]
        A brief message to display at the top left in HTML page containing a TMAP
        visualization. You must specify a valid HTML string. No validation is
        performed. Default message: TMAP chemspace visualization<br/>
        Input file: <InfileName><br/>Number of molecules: <Count>
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To visualize chemspace for SMILES strings present in a column name SMILES in
    input file, mapping a categorical data column on TMAP, writing out LSH forest
    for subsequent use to skip the generation of fingerprints, merging TMAP JS file
    into HTML file, and write out a HTML file containing TMAP visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols Source
          -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example for SMILES strings in column name SMILES in input file
    and write out a HTML file containing TMAP visualization, type:

        % VisualizeChemspaceUsingTMAP.py --colSMILES SMILES
          --categoricalDataCols Source
          -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example for mapping categrorical data in column number 4 in
    input file and write out a HTML file containing TMAP visualization, type:

        % VisualizeChemspaceUsingTMAP.py --colmode colnum
          --categoricalDataCols 4
          -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example for mapping both categrorical and numerical data
    coumns and write out a HTML file containing TMAP visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols "Source"
          --numericalDataCols "MolWt,MolLogP"
          -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example for mapping both categrorical and numerical data
    coumns along with specified colormaps and write out a HTML file containing
    TMAP visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols "Source"
          --categoricalDataColormaps "tab10"
          --numericalDataCols "MolWt,MolLogP"
          --numericalDataColormaps "viridis, plasma"
          -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example for mapping both categrorical and numerical data
    coumns along with displaying specific data under the structure display  and
    write out a HTML file containing TMAP visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols "Source"
          --numericalDataCols "MolWt,NHOHCount,NOCount,MolLogP,
          NumRotatableBonds,TPSA" --structureDisplayDataCols "Name,ID"
          -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example for restoring LSH forest data from a file to skip the
    generation of fingerpritns and write out a HTML file containing TMAP
    visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols Source
           --lshForestFileRestore yes -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example in multiprocessing mode on all available CPUs without
    loading all data into memory and write out  a HTML file containing TMAP
    visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols Source
          --mp yes -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example in multiprocessing mode on all available CPUs by
    loading all data into memory and write out  a HTML file containing TMAP
    visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols Source
          --mp yes --mpParams "inputDataMode,InMemory"
          -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example in multiprocessing mode on specific number of CPUs
    and chunk size without loading all data into memory and write out a HTML file
    containing TMAP visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols Source
          --mp yes --mpParams "inputDataMode,lazy,numProcesses,4,
          chunkSize,50" -i SampleChemspace.csv -o SampleChemspace.html

    To run the first example using a set of specified parameters to generate
    fingerprints and LSH forest, configure faerun and scatter plot layout, and
    write out a HTML file containing TMAP visualization, type:

        % VisualizeChemspaceUsingTMAP.py --categoricalDataCols Source
          --minHashFPParams "radius,3,numPermutations,2048"
          --lshForestParams "dim,2048,numPrefixTrees,128"
          --lshLayoutConfigParams "k,75,kc,20,slRepeats,2,
          slExtraScalingSteps,4,mmmRepeats,2" 
          --faerunConfigParams "clearColor, #000000,thumbnailWidth, 250"
          --faerunScatterPlotParams "shader,circle,pointScale,4"
          --tmapDisplayMsg "TMAP Chemspace visualization"
          -i SampleChemspace.csv -o SampleChemspace.html

Author:
    Manish Sud(msud@san.rr.com)

See also:
    RDKitConvertFileFormat.py, RDKitCalculateMolecularDescriptors.py,
    RDKitStandardizeMolecules.py

Copyright:
    Copyright (C) 2026 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using TMAP and
    Faerun, open source software packages for visualizing chemspace, and
    RDKit, an open source toolkit for cheminformatics developed by Greg
    Landrum.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
