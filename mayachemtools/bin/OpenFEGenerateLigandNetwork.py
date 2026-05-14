#!/usr/bin/env python
#
# File: OpenFEGenerateLigandNetwork.py
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
    GenerateLigandNetwork()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def GenerateLigandNetwork():
    """Generate ligand network for molecules and write them out."""

    # Process ligand molecules...
    Mols = ProcessLigandMolecules()

    # Validate central ligand name for radial network...
    ValidateRadialCentralLigandName(Mols)

    # Initialize atom mappers...
    Mappers = InitializeAtomMappers()

    # Initialize atom scorer...
    MapperScorer = InitializeAtomMapperScorer()

    # Generate network...
    GenerateNetwork(Mols, Mappers, MapperScorer)


def GenerateNetwork(Mols, Mappers, MapperScorer):
    """Generate network and write it out."""

    NetworkName = OptionsInfo["Network"]
    NetworkParams = OptionsInfo["NetworkParams"]

    MiscUtil.PrintInfo("\nChanging directory to %s..." % OptionsInfo["OutfileDir"])
    os.chdir(OptionsInfo["OutfileDirPath"])

    MiscUtil.PrintInfo("\nGenerating ligand network (%s)..." % NetworkName)

    LigandNetwork = OpenFEUtil.GenerateLigandNetwork(Mols, NetworkName, NetworkParams, Mappers, MapperScorer)

    # Write out ligand network graphml and image files...
    WriteLigandNetworkOutputFiles(LigandNetwork, NetworkName)

    #  Write out image files for edges...
    WriteLigandNetworkEdgesOutputFiles(LigandNetwork, NetworkName)


def WriteLigandNetworkOutputFiles(LigandNetwork, NetworkName):
    """Write ligand network output files."""

    GraphMLOutfile, ImageOutfile = SetupLigandNetworkOutputFilePaths(NetworkName)

    # Write graphml file...
    MiscUtil.PrintInfo("Writing %s..." % GraphMLOutfile)
    OpenFEUtil.WriteLigandNetworkGraphMLFile(LigandNetwork, GraphMLOutfile)

    # Write image file...
    MiscUtil.PrintInfo("Writing %s..." % ImageOutfile)
    OpenFEUtil.WriteLigandNetworkImageFile(LigandNetwork, ImageOutfile)


def WriteLigandNetworkEdgesOutputFiles(LigandNetwork, NetworkName):
    """Write ligand network edges output files."""

    if not OptionsInfo["NetworkParams"]["OutputEdges"]:
        return

    NetworkEdges = [Edge for Edge in LigandNetwork.edges]

    if len(NetworkEdges):
        MiscUtil.PrintInfo(
            "Writing %s edge output files <MolName1>_To_<MolName2>_*.png to %s subdirectory..."
            % (len(NetworkEdges), OptionsInfo["EdgesOutfileDir"])
        )

    for Edge in NetworkEdges:
        EdgeOutfile = SetupLigandNetworkEdgeOutputFilePath(Edge, NetworkName)
        OpenFEUtil.WriteMappingImageFile(Edge, EdgeOutfile)


def SetupLigandNetworkOutputFilePaths(NetworkName):
    """Setup ligand network output file paths."""

    NetworkOutfilePrefix = "%s_Network_%s_Mapper_%s" % (OptionsInfo["OutfilePrefix"], NetworkName, SetupMapperLabel())

    GraphMLOutfile = "%s.graphml" % NetworkOutfilePrefix
    ImageOutfile = "%s.%s" % (NetworkOutfilePrefix, OptionsInfo["NetworkParams"]["OutputNetworkFormat"])

    return (GraphMLOutfile, ImageOutfile)


def SetupLigandNetworkEdgeOutputFilePath(Edge, NetworkName):
    """Setup ligand network edge outfile file path."""

    EdgeOutfile = "%s_To_%s_Network_%s_Mapper_%s.png" % (
        Edge.componentA.name,
        Edge.componentB.name,
        NetworkName,
        SetupMapperLabel(),
    )
    EdgeOutfile = re.sub(" ", "_", EdgeOutfile)

    EdgeOutfilePath = os.path.join(OptionsInfo["EdgesOutfileDir"], EdgeOutfile)

    return EdgeOutfilePath


def SetupMapperLabel():
    """Setup mapper label."""

    return "_".join(OptionsInfo["MapperList"])


def InitializeAtomMappers():
    """Initialize atom mappers.."""

    MiscUtil.PrintInfo("\nInitializing atom mappers (%s)..." % " ".join(OptionsInfo["MapperList"]))
    Mappers = OpenFEUtil.InitializeAtomMappers(OptionsInfo["MapperList"], OptionsInfo["MapperParams"])

    return Mappers


def InitializeAtomMapperScorer():
    """Initialize atom mapper scorer."""

    MiscUtil.PrintInfo("\nInitializing atom mapper scorer (%s)..." % OptionsInfo["MapperScorer"])
    MapperScorer = OpenFEUtil.InitializeAtomMapperScorer(OptionsInfo["MapperScorer"])

    return MapperScorer


def ProcessLigandMolecules():
    """Process ligand molecules."""

    MiscUtil.PrintInfo("\nProcessing file %s..." % OptionsInfo["Infile"])
    Mols, MolCount, ValidMolCount = OpenFEUtil.ReadAndValidateMolecules(
        OptionsInfo["InfilePath"], **OptionsInfo["InfileParams"]
    )

    MiscUtil.PrintInfo("\nTotal number of molecules: %d" % MolCount)
    MiscUtil.PrintInfo("Number of valid molecules: %d" % ValidMolCount)
    MiscUtil.PrintInfo("Number of ignored molecules: %d" % (MolCount - ValidMolCount))

    if ValidMolCount == 0:
        MiscUtil.PrintInfo("")
        MiscUtil.PrintError("No valid ligand molecules found in input file.\n")

    return Mols


def ValidateRadialCentralLigandName(Mols):
    """Validate central ligand name for radial network."""

    if not OptionsInfo["RadialNetworkStatus"]:
        return

    if not OpenFEUtil.IsMolNamePresent(Mols, OptionsInfo["NetworkParams"]["RadialCentralLigand"]):
        MiscUtil.PrintError(
            'The value specified, %s, for parameter name, radialCentralLigand, using option "-n, --networkParams" is not valid. Failed to find the specified molecule name in input file.'
            % (OptionsInfo["NetworkParams"]["RadialCentralLigand"])
        )

    if OpenFEUtil.IsMolNamePresentMultipleTimes(Mols, OptionsInfo["NetworkParams"]["RadialCentralLigand"]):
        MiscUtil.PrintError(
            'The value specified, %s, for parameter name, radialCentralLigand, using option "-n, --networkParams" is not valid. Found multiple occurrences of the specified molecule name in input file.'
            % (OptionsInfo["NetworkParams"]["RadialCentralLigand"])
        )


def ProcessOutfilePrefixOption():
    """Process outfile prefix option."""

    OutfilePrefix = Options["--outfilePrefix"]

    if re.match("^auto$", OutfilePrefix, re.I):
        OutfilePrefix = OptionsInfo["InfileRoot"]

    OptionsInfo["OutfilePrefix"] = OutfilePrefix


def ProcessOutfileDirOption():
    """Process outfile directory Option."""

    OutfileDir = Options["--outfileDir"]
    OutfileDirPath = os.path.abspath(OutfileDir)

    if not os.path.exists(OutfileDir):
        MiscUtil.PrintInfo("\nCreating output directory %s..." % (OutfileDir))
        os.mkdir(OutfileDirPath)

    OptionsInfo["OutfileDir"] = OutfileDir
    OptionsInfo["OutfileDirPath"] = OutfileDirPath

    # Setup edges output directory...
    EdgesOutfileDir = "EdgeImages"
    EdgesOutfileDirPath = os.path.join(OptionsInfo["OutfileDirPath"], EdgesOutfileDir)
    if OptionsInfo["NetworkParams"]["OutputEdges"]:
        if not os.path.exists(EdgesOutfileDirPath):
            os.mkdir(EdgesOutfileDirPath)

    OptionsInfo["EdgesOutfileDir"] = EdgesOutfileDir
    OptionsInfo["EdgesOutfileDirPath"] = EdgesOutfileDirPath


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

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["Infile"])
    OptionsInfo["InfileRoot"] = FileName

    OptionsInfo["LoggingLevel"] = Options["--loggingLevel"]

    ParamsDefaultInfoOverride = {"RemoveHydrogens": False}
    OptionsInfo["InfileParams"] = MiscUtil.ProcessOptionInfileParameters(
        "--infileParams",
        Options["--infileParams"],
        InfileName=Options["--infile"],
        ParamsDefaultInfo=ParamsDefaultInfoOverride,
    )

    OptionsInfo["MapperList"] = OpenFEUtil.ProcessOptionOpenFEMapper("-m, --mapper", Options["--mapper"])
    OptionsInfo["MapperParams"] = OpenFEUtil.ProcessOptionOpenFEMapperParameters(
        "-m, --mapperParams", Options["--mapperParams"]
    )
    OptionsInfo["MapperScorer"] = Options["--mapperScorer"]

    OptionsInfo["Network"] = OpenFEUtil.ProcessOptionOpenFENetwork("-n, --network", Options["--network"])
    OptionsInfo["RadialNetworkStatus"] = True if re.match("^Radial$", OptionsInfo["Network"], re.I) else False
    OptionsInfo["NetworkParams"] = OpenFEUtil.ProcessOptionOpenFENetworkParameters(
        "--networkParams", Options["--networkParams"], RadialNetworkStatus=OptionsInfo["RadialNetworkStatus"]
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

    MiscUtil.ValidateOptionTextValue(" --loggingLevel", Options["--loggingLevel"], "Info Warning")

    for Mapper in Options["--mapper"].split(","):
        Mapper = Mapper.strip()
        MiscUtil.ValidateOptionTextValue("-m, --mapper", Mapper, "LOMAP Kartograf")

    MiscUtil.ValidateOptionTextValue("--mapperScorer", Options["--mapperScorer"], "LOMAP")

    MiscUtil.ValidateOptionTextValue("-n, --network", Options["--network"], "LOMAP MinimalSpanning Radial")


# Setup a usage string for docopt...
_docoptUsage_ = """
OpenFEGenerateLigandNetwork.py - Generate ligand network

Usage:
    OpenFEGenerateLigandNetwork.py [--infileParams <Name,Value,...>] [--loggingLevel <Info or Warning>]
                                    [--mapper <mapper1, mapper2,...>] [--mapperParams <Name,Value,..>] [--mapperScorer <LOMAP>]
                                    [--network <text>] [--networkParams <Name,Value,..>] [--outfilePrefix <text>]
                                    [--overwrite] [-w <dir>] -i <infile> -o <outifiledir>
    OpenFEGenerateLigandNetwork.py -h | --help | -e | --examples

Description:
    Generate a ligand network for molecules in an input file and write it out
    as graphml and image files under the output directory. You may optionally
    generate image files for all edges in a ligand network.

    You must specify a valid input file containing 3D coordinates for all
    molecules. In addition, the hydrogens must be present for all molecules
    in the input file.

    The supported input file format is:  SD (.sdf, .sd)

    The supported output file formats are:  GraphML (.graphml), SVG (.svg),
    PNG (.png), etc.

    Possible output directories:
        
        <OutfileDir>
        <OutfileDir>/Edges
        
    Possible outfile prefixe:
        
        <OutfilePrefix> or <InfileRoot>
        
    Possible output files:
        
        <OutfilePrefix>_Network_<NetworkName>_Mapper_<MapperNames>.graphml
        <OutfilePrefix>_Network_<NetworkName>_Mapper_<MapperNames>.<ImgExt>
        
        Edges:
        
        <MolName1>_To_<MolName2>_<NetworkName>_Mapper_<MapperNames>.png
         ... ... ...

Options:
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
            
            SD: removeHydrogens,no,sanitize,yes,strictParsing,yes
            
    --loggingLevel <Info or Warning>  [default: Warning]
        Logging level to configure the 'root logger' via logging.basicConfig()
        function. The default logging level is changed from 'logging.INFO' to
        'logging.WARNING'. Otherwise, OpenFE and its associated modules
        may generate a lot of informational messages.
    -m, --mapper <mapper1, mapper2>  [default: LOMAP]
        A comma delimited names of atom mappers to use for generating a ligand
        network. Possible values: LOMAP [ Lead Optimization MAPer; Ref 176 ] or
        Kartograf [ Ref 177 ]. You may specify multiple mappers for generating
        mapping between two molecules. All specified mappers are employed to
        identify the highest scoring edges for generating a ligand network.
    --mapperParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for atom
        mappers used during the generation of a ligand network.
        
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
        Atom mapper scorer to use for scoring edge mappings during generation
        of a ligand network. Possible value: LOMAP. The atom scorer is not used
        during the generation of MinimalSpanning network.
    -n, --network <text>  [default: MinimalSpanning]
        Name of a ligand network to generate. Possible values: LOMAP,
        MinimalSpanning or Radial. 
    --networkParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        a ligand network.
        
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
                must be specified to generate a radial ligand network.
            
            outputEdges: Generate PNG image files for all edges in a ligand
                network.
            outputNetworkFormat: Valid image file format for ligand network.
                You must specify a valid format supported by Python module
                Matplotlib. For example: PNG (.png), SVG (.svg), PDF (.pdf),
                etc. In addition, the graphml file is always generated.
            
    -o, --outfileDir <outfiledir>
        Output directory.
    --outfilePrefix <text>  [default: auto]
        Prefix for generating output files under output directory.
    --overwrite
        Overwrite existing files.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To generate a minimal spanning ligand network for molecules in a SD file with
    3D structures, employing LOMAP atom mapper to map egdes, and write out
    network GraphML, SVG and edge image files to output directory, type:

        % OpenFEGenerateLigandNetwork.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsMSTNetwork

    To run the first example along with generating PNG image files for all edges,
    type:

        % OpenFEGenerateLigandNetwork.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsMSTNetwork --networkParams "outputEdges, yes"

    To run the first example employing Kartograf atom mapper to map edges,
    and write out various output files under to directory, type:

        % OpenFEGenerateLigandNetwork.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsMSTNetwork -m Kartograf

    To run the first example for generating a radial ligand network centered
    around the ligand lig_ejm_31, and write out various output files to output
    directory, type:

        % OpenFEGenerateLigandNetwork.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsRadialNetwork -n Radial --networkParams
          "RadialCentralLigand,lig_ejm_31"

    To run the previous example along with generating PNG image files for all edges,
    type:

        % OpenFEGenerateLigandNetwork.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsRadialNetwork -n Radial --networkParams
          "RadialCentralLigand,lig_ejm_31,outputEdges, yes"

    To run the first example by specifying explicit values for various parameters
    and write out various output files to output directory, type:

        % OpenFEGenerateLigandNetwork.py -i SampleTyk2Ligands.sdf
          -o SampleTyk2LigandsLOMAPNetwork
          --loggingLevel Warning -m LOMAP  --mapperParams "lomapTime, 20,
          lomapThreeD, yes, lomapElementChange, yes, lomapElementChange, yes,
          lomapSeed, None, lomapShift, no" -n LOMAP --networkParams
          "lomapDistanceCutoff, 0.4, lomapMaxPathLength, 6,
          lomapRequireCycleCovering, yes, outputEdges, yes"

Author:
    Manish Sud(msud@san.rr.com)

See also:
    OpenFECalculateAbsoluteHydrationFreeEnergy.py,
    OpenFECalculateRelativeBindingFreeEnergy.py,
    OpenFECalculateRelativeHydrationFreeEnergy.py,
    OpenFECalculatePartialCharges.py

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
