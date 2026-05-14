#!/usr/bin/env python
#
# File: OpenMMExecuteMDSimulationProtocol.py
# Author: Manish Sud <msud@san.rr.com>
#
# Acknowledgment: Paul Charifson
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


# Add local python path to the global path and import standard library modules...
import os
import sys
import time
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# OpenMM imports...
try:
    import openmm as mm
    import openmm.app
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
        "\n%s (OpenMM v%s; MayaChemTools v%s; %s): Starting...\n"
        % (ScriptName, mm.Platform.getOpenMMVersion(), MiscUtil.GetMayaChemToolsVersion(), time.asctime())
    )

    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()

    # Retrieve command line arguments and options...
    RetrieveOptions()

    # Process and validate command line arguments and options...
    ProcessOptions()

    # Perform actions required by the script...
    ExecuteMDSimulationProtocol()

    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))


def ExecuteMDSimulationProtocol():
    """Execute MD simulation protocol."""

    # Prepare system for simulation...
    System, Topology, Positions = PrepareSystem()

    # Freeze and restraint atoms...
    FreezeRestraintAtoms(System, Topology, Positions)

    # Setup integrator...
    Integrator = SetupIntegrator()

    # Setup simulation...
    Simulation = SetupSimulation(System, Integrator, Topology, Positions)

    # Write setup files...
    WriteSimulationSetupFiles(System, Integrator)

    # Perform minimization...
    PerformMinimization(Simulation)

    # Set up intial velocities...
    SetupInitialVelocities(Simulation)

    # Execute MD protocol workflow...
    ExecuteMDSimulationProtocolWorkflow(System, Simulation, Integrator)

    # Save final state files...
    WriteFinalStateFiles(Simulation)

    # Reimage and realign trajectory for periodic systems...
    ProcessTrajectory(System, Topology)

    # Fix column name in data log file..
    ProcessDataLogFile()

    # Generate plots using data in log file...
    GeneratePlots()


def PrepareSystem():
    """Prepare system for simulation."""

    System, Topology, Positions = OpenMMUtil.InitializeSystem(
        OptionsInfo["Infile"],
        OptionsInfo["ForcefieldParams"],
        OptionsInfo["SystemParams"],
        OptionsInfo["WaterBox"],
        OptionsInfo["WaterBoxParams"],
        OptionsInfo["SmallMolFile"],
        OptionsInfo["SmallMolID"],
    )

    if OptionsInfo["MDProtocolParams"]["Phase4"] or OptionsInfo["MDProtocolParams"]["Phase5"]:
        if not OpenMMUtil.DoesSystemUsesPeriodicBoundaryConditions(System):
            MiscUtil.PrintInfo("")
            MiscUtil.PrintWarning(
                "A barostat is required for NPT equilibration and production simulations during phase 4 and 5. It appears that your system is a non-periodic system and OpenMM may fail during the addition of a barostat for phase 4 and 5. You must specify a periodic system or add water box to automatically set up a periodic system. "
            )

    MiscUtil.PrintInfo("\nChanging directory to %s..." % OptionsInfo["OutfileDir"])
    os.chdir(OptionsInfo["OutfileDirPath"])

    # Write out a PDB file for the system...
    PDBFile = OptionsInfo["PDBOutfile"]
    MiscUtil.PrintInfo("\nWriting PDB file %s..." % PDBFile)
    OpenMMUtil.WritePDBFile(PDBFile, Topology, Positions, OptionsInfo["OutputParams"]["PDBOutKeepIDs"])

    return (System, Topology, Positions)


def SetupIntegrator():
    """Setup integrator."""

    Integrator = OpenMMUtil.InitializeIntegrator(
        OptionsInfo["IntegratorParams"], OptionsInfo["SystemParams"]["ConstraintErrorTolerance"]
    )

    return Integrator


def SetupSimulation(System, Integrator, Topology, Positions):
    """Setup simulation."""

    Simulation = OpenMMUtil.InitializeSimulation(System, Integrator, Topology, Positions, OptionsInfo["PlatformParams"])

    return Simulation


def SetupInitialVelocities(Simulation):
    """Setup initial velocities."""

    # Set velocities to random values choosen from a Boltzman distribution at a given
    # temperature...
    MDProtocolParams = OptionsInfo["MDProtocolParams"]
    MDProtocolParamsInfo = OpenMMUtil.SetupMDProtocolParameters(OptionsInfo["MDProtocolParams"])

    TemperatureParamName = "Phase1InitialStart" if MDProtocolParams["Phase1"] else "Phase1InitialEnd"

    MiscUtil.PrintInfo("\nSetting initial velocities to temperature (%s K)..." % MDProtocolParams[TemperatureParamName])
    Simulation.context.setVelocitiesToTemperature(MDProtocolParamsInfo[TemperatureParamName])


def PerformMinimization(Simulation):
    """Perform minimization."""

    SimulationParams = OpenMMUtil.SetupSimulationParameters(OptionsInfo["SimulationParams"])

    if not SimulationParams["Minimization"]:
        MiscUtil.PrintInfo("\nSkipping energy minimization...")
        return

    OutputParams = OptionsInfo["OutputParams"]

    # Setup a local minimization reporter...
    MinimizeReporter = None
    if OutputParams["MinimizationDataStdout"] or OutputParams["MinimizationDataLog"]:
        MinimizeReporter = LocalMinimizationReporter()

    if MinimizeReporter is not None:
        MiscUtil.PrintInfo("\nAdding minimization reporters...")
        if OutputParams["MinimizationDataLog"]:
            MiscUtil.PrintInfo(
                "Adding data log minimization reporter (Steps: %s; File: %s)..."
                % (OutputParams["MinimizationDataSteps"], OutputParams["MinimizationDataLogFile"])
            )
        if OutputParams["MinimizationDataStdout"]:
            MiscUtil.PrintInfo(
                "Adding data stdout minimization reporter (Steps: %s)..." % (OutputParams["MinimizationDataSteps"])
            )
    else:
        MiscUtil.PrintInfo("\nSkipping addition of minimization reporters...")

    MaxSteps = SimulationParams["MinimizationMaxSteps"]

    MaxStepsMsg = "MaxSteps: %s" % ("UntilConverged" if MaxSteps == 0 else MaxSteps)
    ToleranceMsg = "Tolerance: %.2f kcal/mol/A (%.2f kjoules/mol/nm)" % (
        SimulationParams["MinimizationToleranceInKcal"],
        SimulationParams["MinimizationToleranceInJoules"],
    )

    MiscUtil.PrintInfo("\nPerforming energy minimization (%s; %s)..." % (MaxStepsMsg, ToleranceMsg))

    if OutputParams["MinimizationDataStdout"]:
        HeaderLine = SetupMinimizationDataOutHeaderLine()
        print("\n%s" % HeaderLine)

    Simulation.minimizeEnergy(
        tolerance=SimulationParams["MinimizationTolerance"], maxIterations=MaxSteps, reporter=MinimizeReporter
    )

    if OutputParams["MinimizationDataLog"]:
        WriteMinimizationDataLogFile(MinimizeReporter.DataOutValues)

    if OutputParams["PDBOutMinimized"]:
        MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["MinimizedPDBOutfile"])
        OpenMMUtil.WriteSimulationStatePDBFile(
            Simulation, OptionsInfo["MinimizedPDBOutfile"], OutputParams["PDBOutKeepIDs"]
        )


def ExecuteMDSimulationProtocolWorkflow(System, Simulation, Integrator):
    """Execute MD simulation prorocol workflow."""

    MiscUtil.PrintInfo("\nExecuting MD simulation protocol...")

    if OptionsInfo["OutputReportersModeAllPhases"]:
        SetupReporters(Simulation)

    TotalSimulationSteps = 0

    Phase1SimulationSteps = PerformPhase1InitialHeating(Simulation, Integrator)
    TotalSimulationSteps += Phase1SimulationSteps

    Phase2SimulationSteps = PerformPhase2HeatingAndCooling(Simulation, Integrator)
    TotalSimulationSteps += Phase2SimulationSteps

    Phase3SimulationSteps = PerformPhase3Equilibration(Simulation, Integrator)
    TotalSimulationSteps += Phase3SimulationSteps

    if OptionsInfo["MDProtocolParams"]["Phase4"] or OptionsInfo["MDProtocolParams"]["Phase5"]:
        Barostat = OpenMMUtil.InitializeBarostat(OptionsInfo["IntegratorParams"])
        MiscUtil.PrintInfo("Adding barostat for NPT simulation...")
        try:
            System.addForce(Barostat)
            Simulation.context.reinitialize(preserveState=True)
        except Exception as ErrMsg:
            MiscUtil.PrintInfo("")
            MiscUtil.PrintError("Failed to add barostat:\n%s\n" % (ErrMsg))

    Phase4SimulationSteps = PerformPhase4Equilibration(Simulation, Integrator)
    TotalSimulationSteps += Phase4SimulationSteps

    if OptionsInfo["MDProtocolParams"]["Phase5"]:
        if not OptionsInfo["OutputReportersModeAllPhases"]:
            SetupReporters(Simulation)

    Phase5SimulationSteps = PerformPhase5ProductionRun(Simulation, Integrator)
    TotalSimulationSteps += Phase5SimulationSteps

    MiscUtil.PrintInfo(
        "\nFinishing executing MD protocol (TotalSteps: %s; TotalTime: %s)..."
        % (TotalSimulationSteps, GetTotalSimulationTime(TotalSimulationSteps))
    )


def PerformPhase1InitialHeating(Simulation, Integrator):
    """Perform phase 1 NVT initial heating."""

    if not OptionsInfo["MDProtocolParams"]["Phase1"]:
        MiscUtil.PrintInfo("\nSkipping phase 1 initial heating...")
        return 0

    MDProtocolParams = OptionsInfo["MDProtocolParams"]
    OutputParams = OptionsInfo["OutputParams"]

    # Perform intial heating along with equilibration...
    InitialStart = MDProtocolParams["Phase1InitialStart"]
    InitialEnd = MDProtocolParams["Phase1InitialEnd"]
    InitialChange = MDProtocolParams["Phase1InitialChange"]
    InitialSteps = MDProtocolParams["Phase1InitialSteps"]

    Barostat = None
    TotalSimulationSteps = 0

    MiscUtil.PrintInfo(
        "\nPerforming phase 1 initial heating (Ensemble: NVT; Start: %.1f K; End: %.1f K; Change: %.1f K)..."
        % (InitialStart, InitialEnd, InitialChange)
    )
    TotalInitialSimulationSteps = OpenMMUtil.PerformAnnealing(
        Simulation, Integrator, Barostat, InitialStart, InitialEnd, InitialChange, InitialSteps
    )
    MiscUtil.PrintInfo(
        "Finished initial heating (TotalSteps: %s; TotalTime: %s)..."
        % (TotalInitialSimulationSteps, GetTotalSimulationTime(TotalInitialSimulationSteps))
    )
    TotalSimulationSteps += TotalInitialSimulationSteps

    # Perform equilibration after intial heating...
    InitialEquilibrationSteps = MDProtocolParams["Phase1InitialEquilibrationSteps"]
    MiscUtil.PrintInfo(
        "\nPerforming phase 1 equilibration after initial heating (Ensemble: NVT; Steps: %s; Time: %s)..."
        % (InitialEquilibrationSteps, GetTotalSimulationTime(InitialEquilibrationSteps))
    )
    Simulation.step(InitialEquilibrationSteps)
    TotalSimulationSteps += InitialEquilibrationSteps

    if OutputParams["PDBOutPhase1HeatedNVT"]:
        MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["Phase1HeatedNVTPDBOutfile"])
        OpenMMUtil.WriteSimulationStatePDBFile(
            Simulation, OptionsInfo["Phase1HeatedNVTPDBOutfile"], OutputParams["PDBOutKeepIDs"]
        )

    return TotalSimulationSteps


def PerformPhase2HeatingAndCooling(Simulation, Integrator):
    """Perform phase 2 NVT heating and cooling."""

    if not OptionsInfo["MDProtocolParams"]["Phase2"]:
        MiscUtil.PrintInfo("\nSkipping phase 2 heating and cooling...")
        return 0

    MDProtocolParams = OptionsInfo["MDProtocolParams"]
    OutputParams = OptionsInfo["OutputParams"]

    # Peform heating and coolling annealing cycles along with equilibration...
    Cycles = MDProtocolParams["Phase2Cycles"]
    CycleStart = MDProtocolParams["Phase2CycleStart"]
    CycleEnd = MDProtocolParams["Phase2CycleEnd"]
    CycleChange = MDProtocolParams["Phase2CycleChange"]
    CycleSteps = MDProtocolParams["Phase2CycleSteps"]
    CycleEquilibrationSteps = MDProtocolParams["Phase2CycleEquilibrationSteps"]

    Barostat = None
    TotalSimulationSteps = 0

    MiscUtil.PrintInfo("\nPerforming phase 2 heating and cooling cycles (NumCycles: %s)..." % (Cycles))
    for Cycle in range(Cycles):
        MiscUtil.PrintInfo("\nPerforming heating and cooling cycle %s..." % (Cycle + 1))

        MiscUtil.PrintInfo(
            "\nPerforming heating (Start: %.1f K; End: %.1f K; Change: %.1f K)..." % (CycleStart, CycleEnd, CycleChange)
        )
        TotalCycleSimulationSteps = OpenMMUtil.PerformAnnealing(
            Simulation, Integrator, Barostat, CycleStart, CycleEnd, CycleChange, CycleSteps
        )
        MiscUtil.PrintInfo(
            "Finished heating cycle (TotalSteps: %s; TotalTime: %s)..."
            % (TotalCycleSimulationSteps, GetTotalSimulationTime(TotalCycleSimulationSteps))
        )
        TotalSimulationSteps += TotalCycleSimulationSteps

        MiscUtil.PrintInfo(
            "\nPerforming equilibration (Steps: %s; Time: %s)..."
            % (CycleEquilibrationSteps, GetTotalSimulationTime(CycleEquilibrationSteps))
        )
        Simulation.step(CycleEquilibrationSteps)
        TotalSimulationSteps += CycleEquilibrationSteps

        MiscUtil.PrintInfo(
            "\nPerforming cooling (Start: %.1f K; End: %.1f K; Change: %.1f K)..." % (CycleEnd, CycleStart, CycleChange)
        )
        TotalCycleSimulationSteps = OpenMMUtil.PerformAnnealing(
            Simulation, Integrator, Barostat, CycleEnd, CycleStart, CycleChange, CycleSteps
        )
        MiscUtil.PrintInfo(
            "Finished cooling cycle (TotalSteps: %s; TotalTime: %s)..."
            % (TotalCycleSimulationSteps, GetTotalSimulationTime(TotalCycleSimulationSteps))
        )
        TotalSimulationSteps += TotalCycleSimulationSteps

        MiscUtil.PrintInfo(
            "\nPerforming equilibration (Steps: %s; Time: %s)..."
            % (CycleEquilibrationSteps, GetTotalSimulationTime(CycleEquilibrationSteps))
        )
        Simulation.step(CycleEquilibrationSteps)
        TotalSimulationSteps += CycleEquilibrationSteps

        MiscUtil.PrintInfo("\nFinished heating and cooling cycle %s..." % (Cycle + 1))

    if OutputParams["PDBOutPhase2AnnealedNVT"]:
        MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["Phase2AnnealedNVTPDBOutfile"])
        OpenMMUtil.WriteSimulationStatePDBFile(
            Simulation, OptionsInfo["Phase2AnnealedNVTPDBOutfile"], OutputParams["PDBOutKeepIDs"]
        )

    return TotalSimulationSteps


def PerformPhase3Equilibration(Simulation, Integrator):
    """Perform phase 3 NVT equilibration."""

    if not OptionsInfo["MDProtocolParams"]["Phase3"]:
        MiscUtil.PrintInfo("\nSkipping phase 3 equilibration...")
        return 0

    MDProtocolParams = OptionsInfo["MDProtocolParams"]
    OutputParams = OptionsInfo["OutputParams"]

    Phase3Steps = MDProtocolParams["Phase3Steps"]
    MiscUtil.PrintInfo(
        "\nPerforming phase 3 equilibration (Ensemble: NVT; Steps: %s; Time: %s)..."
        % (Phase3Steps, GetTotalSimulationTime(Phase3Steps))
    )
    Simulation.step(Phase3Steps)

    if OutputParams["PDBOutPhase3EquilibratedNVT"]:
        MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["Phase3EquilibratedNVTPDBOutfile"])
        OpenMMUtil.WriteSimulationStatePDBFile(
            Simulation, OptionsInfo["Phase3EquilibratedNVTPDBOutfile"], OutputParams["PDBOutKeepIDs"]
        )

    return Phase3Steps


def PerformPhase4Equilibration(Simulation, Integrator):
    """Perform phase 4 NPT equilibration."""

    if not OptionsInfo["MDProtocolParams"]["Phase4"]:
        MiscUtil.PrintInfo("\nSkipping phase 4 equilibration...")
        return 0

    MDProtocolParams = OptionsInfo["MDProtocolParams"]
    OutputParams = OptionsInfo["OutputParams"]

    Phase4Steps = MDProtocolParams["Phase4Steps"]
    MiscUtil.PrintInfo(
        "\nPerforming phase 4 equilibration (Ensemble: NPT; Steps: %s; Time: %s)..."
        % (Phase4Steps, GetTotalSimulationTime(Phase4Steps))
    )
    Simulation.step(Phase4Steps)

    if OutputParams["PDBOutPhase4EquilibratedNPT"]:
        MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["Phase4EquilibratedNPTPDBOutfile"])
        OpenMMUtil.WriteSimulationStatePDBFile(
            Simulation, OptionsInfo["Phase4EquilibratedNPTPDBOutfile"], OutputParams["PDBOutKeepIDs"]
        )

    return Phase4Steps


def PerformPhase5ProductionRun(Simulation, Integrator):
    """Perform phase 5 NPT production run.."""

    if not OptionsInfo["MDProtocolParams"]["Phase5"]:
        MiscUtil.PrintInfo("\nSkipping phase 5 equilibration...")
        return 0

    MDProtocolParamsInfo = OpenMMUtil.SetupMDProtocolParameters(OptionsInfo["MDProtocolParams"])
    OutputParams = OptionsInfo["OutputParams"]

    Phase5Steps = MDProtocolParamsInfo["Phase5Steps"]
    Phase5StepSize = MDProtocolParamsInfo["Phase5StepSize"]
    if Phase5StepSize is not None:
        # Setup step size for phase 5 simulation..
        MiscUtil.PrintInfo("\nModifying step size for phase 5 (StepSize: %s)..." % Phase5StepSize)
        Integrator.setStepSize(Phase5StepSize)

    MiscUtil.PrintInfo(
        "\nPerforming phase 5 production run (Ensemble: NPT; Steps: %s;  Time: %s)..."
        % (Phase5Steps, GetTotalSimulationTime(Phase5Steps, Phase5StepSize))
    )
    Simulation.step(Phase5Steps)

    if OutputParams["PDBOutPhase5ProductionNPT"]:
        MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["Phase5ProductionNPTPDBOutfile"])
        OpenMMUtil.WriteSimulationStatePDBFile(
            Simulation, OptionsInfo["Phase5ProductionNPTPDBOutfile"], OutputParams["PDBOutKeepIDs"]
        )

    return Phase5Steps


def GetTotalSimulationTime(SimulationSteps, StepSize=None):
    """Get total simulation time."""

    if StepSize is None:
        IntegratorParamsInfo = OpenMMUtil.SetupIntegratorParameters(OptionsInfo["IntegratorParams"])
        StepSize = IntegratorParamsInfo["StepSize"]

    TotalTime = OpenMMUtil.GetFormattedTotalSimulationTime(StepSize, SimulationSteps)

    return TotalTime


def SetupReporters(Simulation):
    """Setup reporters."""

    DataAppend = False
    (TrajReporter, DataLogReporter, DataStdoutReporter, CheckpointReporter) = OpenMMUtil.InitializeReporters(
        OptionsInfo["OutputParams"], OptionsInfo["SimulationParams"]["Steps"], DataAppend
    )

    if TrajReporter is None and DataLogReporter is None and DataStdoutReporter is None and CheckpointReporter is None:
        MiscUtil.PrintInfo("\nSkip adding  reporters...")
        return

    MiscUtil.PrintInfo("\nAdding reporters...")

    OutputParams = OptionsInfo["OutputParams"]
    AppendMsg = ""
    if TrajReporter is not None:
        MiscUtil.PrintInfo(
            "Adding trajectory reporter (Steps: %s; File: %s%s)..."
            % (OutputParams["TrajSteps"], OutputParams["TrajFile"], AppendMsg)
        )
        Simulation.reporters.append(TrajReporter)

    if CheckpointReporter is not None:
        MiscUtil.PrintInfo(
            "Adding checkpoint reporter (Steps: %s; File: %s)..."
            % (OutputParams["CheckpointSteps"], OutputParams["CheckpointFile"])
        )
        Simulation.reporters.append(CheckpointReporter)

    if DataLogReporter is not None:
        MiscUtil.PrintInfo(
            "Adding data log reporter (Steps: %s; File: %s%s)..."
            % (OutputParams["DataLogSteps"], OutputParams["DataLogFile"], AppendMsg)
        )
        Simulation.reporters.append(DataLogReporter)

    if DataStdoutReporter is not None:
        MiscUtil.PrintInfo("Adding data stdout reporter (Steps: %s)..." % (OutputParams["DataStdoutSteps"]))
        Simulation.reporters.append(DataStdoutReporter)


class LocalMinimizationReporter(mm.MinimizationReporter):
    """Setup a local minimization reporter."""

    (DataSteps, DataOutTypeList, DataOutDelimiter, StdoutStatus) = [None] * 4

    DataOutValues = []
    First = True

    def report(self, Iteration, PositonsList, GradientsList, DataStatisticsMap):
        """Report and track minimization."""

        if self.First:
            # Initialize...
            self.DataSteps = OptionsInfo["OutputParams"]["MinimizationDataSteps"]
            self.DataOutTypeList = OptionsInfo["OutputParams"]["MinimizationDataOutTypeOpenMMNameList"]
            self.DataOutDelimiter = OptionsInfo["OutputParams"]["DataOutDelimiter"]
            self.StdoutStatus = True if OptionsInfo["OutputParams"]["MinimizationDataStdout"] else False

            self.First = False

        if Iteration % self.DataSteps == 0:
            # Setup data values...
            DataValues = []
            DataValues.append("%s" % Iteration)
            for DataType in self.DataOutTypeList:
                DataValue = "%.4f" % DataStatisticsMap[DataType]
                DataValues.append(DataValue)

            # Track data...
            self.DataOutValues.append(DataValues)

            # Print data values...
            if self.StdoutStatus:
                print("%s" % self.DataOutDelimiter.join(DataValues))

        # This method must return a bool. You may return true for early termination.
        return False


def WriteMinimizationDataLogFile(DataOutValues):
    """Write minimization data log file."""

    OutputParams = OptionsInfo["OutputParams"]

    Outfile = OutputParams["MinimizationDataLogFile"]
    OutDelimiter = OutputParams["DataOutDelimiter"]

    MiscUtil.PrintInfo("\nWriting minimization log file %s..." % Outfile)
    OutFH = open(Outfile, "w")

    HeaderLine = SetupMinimizationDataOutHeaderLine()
    OutFH.write("%s\n" % HeaderLine)

    for LineWords in DataOutValues:
        Line = OutDelimiter.join(LineWords)
        OutFH.write("%s\n" % Line)

    OutFH.close()


def SetupMinimizationDataOutHeaderLine():
    """Setup minimization data output header line."""

    LineWords = ["Iteration"]
    for Label in OptionsInfo["OutputParams"]["MinimizationDataOutTypeList"]:
        if re.match("^(SystemEnergy|RestraintEnergy)$", Label, re.I):
            LineWords.append("%s(kjoules/mol)" % Label)
        elif re.match("^RestraintStrength$", Label, re.I):
            LineWords.append("%s(kjoules/mol/nm^2)" % Label)
        else:
            LineWords.append(Label)

    Line = OptionsInfo["OutputParams"]["DataOutDelimiter"].join(LineWords)

    return Line


def FreezeRestraintAtoms(System, Topology, Positions):
    """Handle freezing and restraining of atoms."""

    FreezeAtomList, RestraintAtomList = OpenMMUtil.ValidateAndFreezeRestraintAtoms(
        OptionsInfo["FreezeAtoms"],
        OptionsInfo["FreezeAtomsParams"],
        OptionsInfo["RestraintAtoms"],
        OptionsInfo["RestraintAtomsParams"],
        OptionsInfo["RestraintSpringConstant"],
        OptionsInfo["SystemParams"],
        System,
        Topology,
        Positions,
    )

    #  Check and adjust step size...
    if FreezeAtomList is not None or RestraintAtomList is not None:
        if re.match("^auto$", OptionsInfo["IntegratorParams"]["StepSizeSpecified"], re.I):
            # Automatically set stepSize to 2.0 fs..
            OptionsInfo["IntegratorParams"]["StepSize"] = 2.0
            MiscUtil.PrintInfo("")
            MiscUtil.PrintWarning(
                'The time step has been automatically set to %s fs during freezing or restraining of atoms. You may specify an explicit value for parameter name, stepSize, using "--integratorParams" option.'
                % (OptionsInfo["IntegratorParams"]["StepSize"])
            )
        elif OptionsInfo["IntegratorParams"]["StepSize"] > 2:
            MiscUtil.PrintInfo("")
            MiscUtil.PrintWarning(
                'A word to the wise: The parameter value specified, %s, for parameter name, stepSize, using "--integratorParams" option may be too large. You may want to consider using a smaller time step. Othwerwise, your simulation may blow up.'
                % (OptionsInfo["IntegratorParams"]["StepSize"])
            )
            MiscUtil.PrintInfo("")


def WriteSimulationSetupFiles(System, Integrator):
    """Write simulation setup files for system and integrator."""

    OutputParams = OptionsInfo["OutputParams"]

    if OutputParams["XmlSystemOut"] or OutputParams["XmlIntegratorOut"]:
        MiscUtil.PrintInfo("")

    if OutputParams["XmlSystemOut"]:
        Outfile = OutputParams["XmlSystemFile"]
        MiscUtil.PrintInfo("Writing system setup XML file %s..." % Outfile)
        with open(Outfile, mode="w") as OutFH:
            OutFH.write(mm.XmlSerializer.serialize(System))

    if OutputParams["XmlIntegratorOut"]:
        Outfile = OutputParams["XmlIntegratorFile"]
        MiscUtil.PrintInfo("Writing integrator setup XML file %s..." % Outfile)
        with open(Outfile, mode="w") as OutFH:
            OutFH.write(mm.XmlSerializer.serialize(Integrator))


def WriteFinalStateFiles(Simulation):
    """Write final state files."""

    OutputParams = OptionsInfo["OutputParams"]

    if OutputParams["SaveFinalStateCheckpoint"] or OutputParams["SaveFinalStateXML"] or OutputParams["PDBOutFinal"]:
        MiscUtil.PrintInfo("")

    if OutputParams["SaveFinalStateCheckpoint"]:
        Outfile = OutputParams["SaveFinalStateCheckpointFile"]
        MiscUtil.PrintInfo("Writing final state checkpoint file %s..." % Outfile)
        Simulation.saveCheckpoint(Outfile)

    if OutputParams["SaveFinalStateXML"]:
        Outfile = OutputParams["SaveFinalStateXMLFile"]
        MiscUtil.PrintInfo("Writing final state XML file %s..." % Outfile)
        Simulation.saveState(Outfile)

    if OutputParams["PDBOutFinal"]:
        MiscUtil.PrintInfo("\nWriting PDB file %s..." % OptionsInfo["FinalPDBOutfile"])
        OpenMMUtil.WriteSimulationStatePDBFile(
            Simulation, OptionsInfo["FinalPDBOutfile"], OutputParams["PDBOutKeepIDs"]
        )


def ProcessTrajectory(System, Topology):
    """Reimage and realign trajectory for periodic systems."""

    TrajTopologyFile = OptionsInfo["PDBOutfile"]

    OpenMMUtil.GenerateReimagedRealignedTrajectoryFiles(
        System,
        Topology,
        TrajTopologyFile,
        OptionsInfo["ReimagedPDBOutfile"],
        OptionsInfo["ReimagedTrajOutfile"],
        OptionsInfo["OutputParams"],
        RealignFrames=True,
    )


def ProcessDataLogFile():
    """Process data log file."""

    OutputParams = OptionsInfo["OutputParams"]
    if not OutputParams["DataLog"] or not os.path.exists(OutputParams["DataLogFile"]):
        return

    DataLogFile = OutputParams["DataLogFile"]
    MiscUtil.PrintInfo("\nProcessing data log file %s..." % DataLogFile)

    OpenMMUtil.FixColumNamesLineInDataLogFile(DataLogFile)


def GeneratePlots():
    """Generate plots using data in log file."""

    OutputParams = OptionsInfo["OutputParams"]
    if (
        not OutputParams["DataLog"]
        or not OutputParams["DataOutTypePlot"]
        or not os.path.exists(OutputParams["DataLogFile"])
    ):
        MiscUtil.PrintInfo("\nSkipping generation of plots...")
        return

    MiscUtil.PrintInfo("\nGenerating plots...")
    InitializePlotParameters()

    DataLogFile = OutputParams["DataLogFile"]

    MiscUtil.PrintInfo("Processing file %s..." % DataLogFile)
    DataLogDF = pd.read_csv(DataLogFile, sep=",")
    DataLogColNames = DataLogDF.columns.tolist()

    # Collect data types to plot...
    DataOutTypePlotList = []
    DataOutTypePlotList.append(OutputParams["DataOutTypePlotX"])
    DataOutTypePlotList.extend(OutputParams["DataOutTypePlotYList"])
    DataOutTypePlotColNames = OpenMMUtil.MapDataOutTypePlotToDataLogColumnNames(DataOutTypePlotList, DataLogColNames)

    DataOutTypePlotFiles = OptionsInfo["DataOutTypePlotFiles"]

    for PlotY in OutputParams["DataOutTypePlotYList"]:
        PlotOutFile = DataOutTypePlotFiles[PlotY]

        PlotX = OutputParams["DataOutTypePlotX"]
        PlotXColName = DataOutTypePlotColNames[PlotX]
        PlotYColName = DataOutTypePlotColNames[PlotY]

        if PlotXColName is None or PlotYColName is None:
            MiscUtil.PrintInfo(
                "Skipping generation of plot file %s (Missing %s or %s data column in data log file)..."
                % (PlotOutFile, PlotX, PlotY)
            )
            continue

        PlotXLabel = PlotXColName
        PlotYLabel = PlotYColName

        if OptionsInfo["OutputReportersModeAllPhases"]:
            PlotTitle = "MD Simulation Protocol"
        else:
            PlotTitle = "MD Production Simulation (NPT)"

        GeneratePlotOutFile(PlotOutFile, DataLogDF, PlotXColName, PlotYColName, PlotXLabel, PlotYLabel, PlotTitle)


def GeneratePlotOutFile(PlotOutFile, DataLogDF, PlotXColName, PlotYColName, PlotXLabel, PlotYLabel, PlotTitle):
    """Generate plot out file."""

    OutPlotParams = OptionsInfo["OutPlotParams"]

    MiscUtil.PrintInfo("Generating plot file %s..." % PlotOutFile)

    # Create a new figure...
    plt.figure()

    # Draw plot...
    PlotType = OutPlotParams["Type"]
    if re.match("^line$", PlotType, re.I):
        Axis = sns.lineplot(DataLogDF, x=PlotXColName, y=PlotYColName, legend=False)
    elif re.match("^linepoint$", PlotType, re.I):
        Axis = sns.lineplot(DataLogDF, x=PlotXColName, y=PlotYColName, marker="o", legend=False)
    elif re.match("^scatter$", PlotType, re.I):
        Axis = sns.scatterplot(DataLogDF, x=PlotXColName, y=PlotYColName, legend=False)
    else:
        MiscUtil.PrintError(
            'The value, %s, specified for "type" using option "--outPlotParams" is not supported. Valid plot types: linepoint, scatter or line'
            % (PlotType)
        )

    # Set labels and title...
    Axis.set(xlabel=PlotXLabel, ylabel=PlotYLabel, title=PlotTitle)

    # Save figure...
    plt.savefig(PlotOutFile)

    # Close the plot...
    plt.close()


def InitializePlotParameters():
    """Initialize plot parameters."""

    if OptionsInfo["OutPlotInitialized"]:
        return

    # Initialize seaborn and matplotlib paramaters...
    OptionsInfo["OutPlotInitialized"] = True

    OutPlotParams = OptionsInfo["OutPlotParams"]
    RCParams = {
        "figure.figsize": (OutPlotParams["Width"], OutPlotParams["Height"]),
        "axes.titleweight": OutPlotParams["TitleWeight"],
        "axes.labelweight": OutPlotParams["LabelWeight"],
    }
    sns.set(
        context=OutPlotParams["Context"],
        style=OutPlotParams["Style"],
        palette=OutPlotParams["Palette"],
        font=OutPlotParams["Font"],
        font_scale=OutPlotParams["FontScale"],
        rc=RCParams,
    )


def ProcessOutfilePrefixOption():
    """Process outfile prefix option."""

    OutfilePrefix = Options["--outfilePrefix"]

    if not re.match("^auto$", OutfilePrefix, re.I):
        OptionsInfo["OutfilePrefix"] = OutfilePrefix
        return

    if OptionsInfo["SmallMolFileMode"]:
        OutfilePrefix = "%s_%s_Complex" % (OptionsInfo["InfileRoot"], OptionsInfo["SmallMolFileRoot"])
    else:
        OutfilePrefix = "%s" % (OptionsInfo["InfileRoot"])

    if re.match("^yes$", Options["--waterBox"], re.I):
        OutfilePrefix = "%s_Solvated" % (OutfilePrefix)

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


def ProcessOutfileNames():
    """Process outfile names."""

    OutputParams = OptionsInfo["OutputParams"]

    PDBOutfile = "%s.%s" % (OptionsInfo["OutfilePrefix"], OutputParams["PDBOutfileExt"])
    ReimagedPDBOutfile = "%s_Reimaged.%s" % (OptionsInfo["OutfilePrefix"], OutputParams["PDBOutfileExt"])
    ReimagedTrajOutfile = "%s_Reimaged.%s" % (OptionsInfo["OutfilePrefix"], OutputParams["TrajFileExt"])

    MinimizedPDBOutfile = "%s_Minimized.%s" % (OptionsInfo["OutfilePrefix"], OutputParams["PDBOutfileExt"])
    FinalPDBOutfile = "%s_Final.%s" % (OptionsInfo["OutfilePrefix"], OutputParams["PDBOutfileExt"])

    Phase1HeatedNVTPDBOutfile = "%s_NVT_Phase1_Heated_Equilibrated.%s" % (
        OptionsInfo["OutfilePrefix"],
        OutputParams["PDBOutfileExt"],
    )
    Phase2AnnealedNVTPDBOutfile = "%s_NVT_Phase2_Annealed_Equilibrated.%s" % (
        OptionsInfo["OutfilePrefix"],
        OutputParams["PDBOutfileExt"],
    )
    Phase3EquilibratedNVTPDBOutfile = "%s_NVT_Phase3_Equilibrated.%s" % (
        OptionsInfo["OutfilePrefix"],
        OutputParams["PDBOutfileExt"],
    )
    Phase4EquilibratedNPTPDBOutfile = "%s_NPT_Phase4_Equilibrated.%s" % (
        OptionsInfo["OutfilePrefix"],
        OutputParams["PDBOutfileExt"],
    )
    Phase5ProductionNPTPDBOutfile = "%s_NPT_Phase5_Production.%s" % (
        OptionsInfo["OutfilePrefix"],
        OutputParams["PDBOutfileExt"],
    )

    OptionsInfo["PDBOutfile"] = PDBOutfile
    OptionsInfo["ReimagedPDBOutfile"] = ReimagedPDBOutfile
    OptionsInfo["ReimagedTrajOutfile"] = ReimagedTrajOutfile

    OptionsInfo["MinimizedPDBOutfile"] = MinimizedPDBOutfile
    OptionsInfo["FinalPDBOutfile"] = FinalPDBOutfile

    OptionsInfo["Phase1HeatedNVTPDBOutfile"] = Phase1HeatedNVTPDBOutfile
    OptionsInfo["Phase2AnnealedNVTPDBOutfile"] = Phase2AnnealedNVTPDBOutfile
    OptionsInfo["Phase3EquilibratedNVTPDBOutfile"] = Phase3EquilibratedNVTPDBOutfile
    OptionsInfo["Phase4EquilibratedNPTPDBOutfile"] = Phase4EquilibratedNPTPDBOutfile
    OptionsInfo["Phase5ProductionNPTPDBOutfile"] = Phase5ProductionNPTPDBOutfile

    OutputParams = OptionsInfo["OutputParams"]
    OutPlotParams = OptionsInfo["OutPlotParams"]

    DataOutTypePlotYList = OutputParams["DataOutTypePlotYList"]
    DataOutTypePlotFiles = {}
    for PlotDataType in DataOutTypePlotYList:
        Outfile = "%s_%sPlot.%s" % (OptionsInfo["OutfilePrefix"], PlotDataType, OutPlotParams["OutExt"])
        if not Options["--overwrite"]:
            if os.path.exists(Outfile):
                MiscUtil.PrintError(
                    'The file name, %s, generated using option "--outfilePrefix" already exist. Use option "--ov" or "--overwrite" and try again. '
                    % (Outfile)
                )
        DataOutTypePlotFiles[PlotDataType] = Outfile
    OptionsInfo["DataOutTypePlotFiles"] = DataOutTypePlotFiles


def ProcessWaterBoxParameters():
    """Process water box parameters."""

    OptionsInfo["WaterBox"] = True if re.match("^yes$", Options["--waterBox"], re.I) else False
    OptionsInfo["WaterBoxParams"] = OpenMMUtil.ProcessOptionOpenMMWaterBoxParameters(
        "--waterBoxParams", Options["--waterBoxParams"]
    )

    if OptionsInfo["WaterBox"]:
        if OptionsInfo["ForcefieldParams"]["ImplicitWater"]:
            MiscUtil.PrintInfo("")
            MiscUtil.PrintWarning(
                'The value, %s, specified using option "--waterBox" may not be valid for the combination of biopolymer and water forcefields, %s and %s, specified using "--forcefieldParams". You may consider using a valid combination of biopolymer and water forcefields for explicit water during the addition of a water box.'
                % (
                    Options["--waterBox"],
                    OptionsInfo["ForcefieldParams"]["Biopolymer"],
                    OptionsInfo["ForcefieldParams"]["Water"],
                )
            )


def ProcessOutPlotParameters():
    """Process out plot parameters."""

    DefaultValues = {"Type": "line", "Width": 10.0, "Height": 5.6}
    OptionsInfo["OutPlotParams"] = MiscUtil.ProcessOptionSeabornPlotParameters(
        "--outPlotParams", Options["--outPlotParams"], DefaultValues
    )
    if not re.match("^(linepoint|scatter|Line)$", OptionsInfo["OutPlotParams"]["Type"], re.I):
        MiscUtil.PrintError(
            'The value, %s, specified for "type" using option "--outPlotParams" is not supported. Valid plot types: linepoint, scatter or line'
            % (OptionsInfo["OutPlotParams"]["Type"])
        )

    for PlotParamName in ["XLabel", "YLabel", "Title"]:
        if not re.match("^auto$", OptionsInfo["OutPlotParams"][PlotParamName], re.I):
            MiscUtil.PrintError(
                'The value, %s, specified for "%s" using option "--outPlotParams" is not supported. Valid value: auto'
                % (PlotParamName, OptionsInfo["OutPlotParams"][PlotParamName])
            )

    OptionsInfo["OutPlotInitialized"] = False


def ProcessOptions():
    """Process and validate command line arguments and options."""

    MiscUtil.PrintInfo("Processing options...")

    ValidateOptions()

    OptionsInfo["Infile"] = Options["--infile"]
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["Infile"])
    OptionsInfo["InfileRoot"] = FileName
    OptionsInfo["InfilePath"] = os.path.abspath(OptionsInfo["Infile"])

    SmallMolFile = Options["--smallMolFile"]
    SmallMolID = Options["--smallMolID"]
    SmallMolFilePath = None
    SmallMolFileMode = False
    SmallMolFileRoot = None
    if SmallMolFile is not None:
        FileDir, FileName, FileExt = MiscUtil.ParseFileName(SmallMolFile)
        SmallMolFileRoot = FileName
        SmallMolFileMode = True
        SmallMolFilePath = os.path.abspath(SmallMolFile)

    OptionsInfo["SmallMolFile"] = SmallMolFile
    OptionsInfo["SmallMolFilePath"] = SmallMolFilePath
    OptionsInfo["SmallMolFileRoot"] = SmallMolFileRoot
    OptionsInfo["SmallMolFileMode"] = SmallMolFileMode
    OptionsInfo["SmallMolID"] = SmallMolID.upper()

    ProcessOutfilePrefixOption()
    ProcessOutfileDirOption()

    ParamsDefaultInfoOverride = {"DataOutType": "Step Speed PotentialEnergy Temperature Time Density Volume"}
    ParamsDefaultInfoOverride["DataOutTypePlotX"] = "Time"
    ParamsDefaultInfoOverride["DataOutTypePlotY"] = "PotentialEnergy Temperature Density Volume"
    for ParamName in [
        "PDBOutMinimized",
        "PDBOutFinal",
        "PDBOutPhase1HeatedNVT",
        "PDBOutPhase2AnnealedNVT",
        "PDBOutPhase3EquilibratedNVT",
        "PDBOutPhase4EquilibratedNPT",
        "PDBOutPhase5ProductionNPT",
    ]:
        ParamsDefaultInfoOverride[ParamName] = True
    OptionsInfo["OutputParams"] = OpenMMUtil.ProcessOptionOpenMMOutputParameters(
        "--outputParams", Options["--outputParams"], OptionsInfo["OutfilePrefix"], ParamsDefaultInfoOverride
    )

    ProcessOutPlotParameters()
    ProcessOutfileNames()

    OptionsInfo["MDProtocolParams"] = OpenMMUtil.ProcessOptionOpenMMMDProtocolParameters(
        "-m, --mdProtocolParams", Options["--mdProtocolParams"]
    )

    OptionsInfo["ForcefieldParams"] = OpenMMUtil.ProcessOptionOpenMMForcefieldParameters(
        "--forcefieldParams", Options["--forcefieldParams"]
    )

    OptionsInfo["FreezeAtoms"] = True if re.match("^yes$", Options["--freezeAtoms"], re.I) else False
    if OptionsInfo["FreezeAtoms"]:
        OptionsInfo["FreezeAtomsParams"] = OpenMMUtil.ProcessOptionOpenMMAtomsSelectionParameters(
            "--freezeAtomsParams", Options["--freezeAtomsParams"]
        )
    else:
        OptionsInfo["FreezeAtomsParams"] = None

    OptionsInfo["OutputReportersMode"] = Options["--outputReportersMode"]
    OptionsInfo["OutputReportersModeAllPhases"] = (
        True if re.match("^AllPhases$", Options["--outputReportersMode"], re.I) else False
    )

    ParamsDefaultInfoOverride = {"Name": Options["--platform"], "Threads": 1}
    OptionsInfo["PlatformParams"] = OpenMMUtil.ProcessOptionOpenMMPlatformParameters(
        "--platformParams", Options["--platformParams"], ParamsDefaultInfoOverride
    )

    OptionsInfo["RestraintAtoms"] = True if re.match("^yes$", Options["--restraintAtoms"], re.I) else False
    if OptionsInfo["RestraintAtoms"]:
        OptionsInfo["RestraintAtomsParams"] = OpenMMUtil.ProcessOptionOpenMMAtomsSelectionParameters(
            "--restraintAtomsParams", Options["--restraintAtomsParams"]
        )
    else:
        OptionsInfo["RestraintAtomsParams"] = None
    OptionsInfo["RestraintSpringConstant"] = float(Options["--restraintSpringConstant"])

    OptionsInfo["SystemParams"] = OpenMMUtil.ProcessOptionOpenMMSystemParameters(
        "--systemParams", Options["--systemParams"]
    )

    OptionsInfo["IntegratorParams"] = OpenMMUtil.ProcessOptionOpenMMIntegratorParameters(
        "--integratorParams",
        Options["--integratorParams"],
        HydrogenMassRepartioningStatus=OptionsInfo["SystemParams"]["HydrogenMassRepartioning"],
    )
    if OptionsInfo["MDProtocolParams"]["Phase1"]:
        OptionsInfo["IntegratorParams"]["Temperature"] = OptionsInfo["MDProtocolParams"]["Phase1InitialStart"]
    else:
        OptionsInfo["IntegratorParams"]["Temperature"] = OptionsInfo["MDProtocolParams"]["Phase1InitialEnd"]

    OptionsInfo["SimulationParams"] = OpenMMUtil.ProcessOptionOpenMMSimulationParameters(
        "--simulationParams", Options["--simulationParams"]
    )

    ProcessWaterBoxParameters()

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
    MiscUtil.ValidateOptionFileExt("-i, --infile", Options["--infile"], "pdb cif")

    FileDir, FileName, FileExt = MiscUtil.ParseFileName(Options["--infile"])
    OutfilePrefix = Options["--outfilePrefix"]
    if not re.match("^auto$", OutfilePrefix, re.I):
        if re.match("^(%s)$" % OutfilePrefix, FileName, re.I):
            MiscUtil.PrintError(
                'The value specified, %s, for option "--outfilePrefix" is not valid. You must specify a value different from, %s, the root of infile name.'
                % (OutfilePrefix, FileName)
            )

    if Options["--smallMolFile"] is not None:
        MiscUtil.ValidateOptionFilePath("-l, --smallMolFile", Options["--smallMolFile"])
        MiscUtil.ValidateOptionFileExt("-l, --smallMolFile", Options["--smallMolFile"], "sd sdf")

    SmallMolID = Options["--smallMolID"]
    if len(SmallMolID) != 3:
        MiscUtil.PrintError(
            'The value specified, %s, for option "--smallMolID" is not valid. You must specify a three letter small molecule ID.'
            % (SmallMolID)
        )

    MiscUtil.ValidateOptionDirPath("-o, --outfileDir", Options["--outfileDir"])
    MiscUtil.ValidateOptionsOutputDirOverwrite(
        "-o, --outfileDir", Options["--outfileDir"], "--overwrite", Options["--overwrite"]
    )

    MiscUtil.ValidateOptionTextValue("--freezeAtoms", Options["--freezeAtoms"], "yes no")
    if re.match("^yes$", Options["--freezeAtoms"], re.I):
        if Options["--freezeAtomsParams"] is None:
            MiscUtil.PrintError(
                'No value specified for option "--freezeAtomsParams". You must specify valid values during, yes, value for "--freezeAtoms" option.'
            )

    MiscUtil.ValidateOptionTextValue(
        "--outputReportersMode", Options["--outputReportersMode"], "AllPhases ProductionPhaseOnly"
    )

    MiscUtil.ValidateOptionTextValue("-p, --platform", Options["--platform"], "CPU CUDA OpenCL Reference")

    MiscUtil.ValidateOptionTextValue("--restraintAtoms", Options["--restraintAtoms"], "yes no")
    if re.match("^yes$", Options["--restraintAtoms"], re.I):
        if Options["--restraintAtomsParams"] is None:
            MiscUtil.PrintError(
                'No value specified for option "--restraintAtomsParams". You must specify valid values during, yes, value for "--restraintAtoms" option.'
            )

    MiscUtil.ValidateOptionFloatValue("--restraintSpringConstant", Options["--restraintSpringConstant"], {">": 0})

    MiscUtil.ValidateOptionTextValue("--waterBox", Options["--waterBox"], "yes no")


# Setup a usage string for docopt...
_docoptUsage_ = """
OpenMMExecuteMDSimulationProtocol.py - Execute MD simulation workflow

Usage:
    OpenMMExecuteMDSimulationProtocol.py [--forcefieldParams <Name,Value,..>] [--freezeAtoms <yes or no>]
                                         [--freezeAtomsParams <Name,Value,..>] [--integratorParams <Name,Value,..>]
                                         [--outfilePrefix <text>] [--outputParams <Name,Value,..>] [--outPlotParams <Name,Value,...>]
                                         [--outputReportersMode <text>] [--overwrite] [--platform <text>] [--mdProtocolParams <Name,Value,..>]
                                         [--platformParams <Name,Value,..>] [--restraintAtoms <yes or no>] [--restraintAtomsParams <Name,Value,..>]
                                         [--restraintSpringConstant <number>] [--simulationParams <Name,Value,..>] [--smallMolFile <SmallMolFile>]
                                         [--smallMolID <text>] [--systemParams <Name,Value,..>] [--waterBox <yes or no>]
                                         [--waterBoxParams <Name,Value,..>] [-w <dir>] -i <infile>  -o <outifiledir>
    OpenMMExecuteMDSimulationProtocol.py -h | --help | -e | --examples

Description:
    Perform a MD simulation using a simulation protocol. You may run a simulation
    using a macromolecule or a macromolecule in a complex with small molecule.
    By default, the system is minimized before executing the MD simulation protocol.

    The MD protocol consists of the following steps:
        
        . Initial heating (NVT simulation)
        . Heating and cooling cycles (NVT simulation)
        . NVT equilibration
        . NPT equilibration
        . NPT production
        
    The input file must contain a macromolecule already prepared for simulation.
    The preparation of the macromolecule for a simulation generally involves the
    following: identification and replacement non-standard residues; addition of
    missing residues; addition of missing heavy atoms; addition of missing
    hydrogens; addition of a water box which is optional.

    In addition, the small molecule input file must contain a molecule already
    prepared for simulation. It must contain  appropriate 3D coordinates relative
    to the macromolecule along with no missing hydrogens.

    You may optionally add a water box and freeze/restraint atoms for the
    simulation.

    The restart option is not available in the current script. You may employ
    another script named OpenMMPerformMDSimulation.py to restart the
    simulation using the final checkpoint file generated by the current script.

    By default, the MD protocol is executed for a total of 8.15 ns as shown
    below:
        
        ... ... ...
        MD protocol annealing (StepSize: 4 fs)
        
        Phase1 - Initial heating (NVT simulation):
        
        Initial heating (Start: 0.0 K; End: 300.0 K; Change: 5.0 K)
        TotalSteps: 305,000; TotalTime: 1.22 ns
        
        Equilibration after initial heating (Steps: 100,000; Time: 400.00 ps)
        
        Phase2 - Heating and cooling cycles (NVT simulation):
        
        Heating and cooling cycles (NumCycles: 1)
        
        Heating and cooling cycle 1
        Heating (Start: 300.0 K; End: 315.0 K; Change: 1.0 K)
        TotalSteps: 16,000; TotalTime: 64.00 ps

        Equilibration after heating (Steps: 100,000; Time: 400.00 ps)
        
        Cooling (Start: 315.0 K; End: 300.0 K; Step: 1.0 K)
        TotalSteps: 16,000; TotalTime: 64.00 ps
        
        Equilibration after cooling (Steps: 100,000; Time: 400.00 ps)
        
        Phase3 - NVT equilibration: (Steps: 200,000; Time: 800.00 ps)
        
        Phase4 - NPT equilibration: (Steps: 200,000; Time: 800.00 ps)
        
        Phase5 - NPT production: (Steps: 1,000.000; Time: 4.00 ns)
        
        MD protocol Summary: (TotalSteps: 2,037,000; TotalTime: 8.15 ns)
        
        ... ... ...

    The supported macromolecule input file formats are:  PDB (.pdb) and
    CIF (.cif)

    The supported small molecule input file format are : SD (.sdf, .sd)

    Possible outfile prefixes:
        
        <InfileRoot>
        <InfileRoot>_Solvated
        <InfileRoot>_<SmallMolFileRoot>
        <InfileRoot>_<SmallMolFileRoot>_Complex_Solvated
        
    Possible output files:

        <OutfilePrefix>.<pdb or cif> [ Initial sytem ]
        <OutfilePrefix>.<dcd or xtc>
        
        <OutfilePrefix>_Reimaged.<pdb or cif> [ First frame ]
        <OutfilePrefix>_Reimaged.<dcd or xtc>
        
        <OutfilePrefix>_Minimized.<pdb or cif>
        <OutfilePrefix>_Final.<pdb or cif>
        
        <OutfilePrefix>_NVT_Phase1_Heated_Equilibated.<pdb or cif>
        <OutfilePrefix>_NVT_Phase2_Annealed_Equilibrated.<pdb or cif>
        <OutfilePrefix>_NVT_Phase3_Equilibrated.<pdb or cif>
        <OutfilePrefix>_NPT_Phase4_Equilibrated.<pdb or cif>
        <OutfilePrefix>_NPT_Phase5_Production.<pdb or cif>
         
        <OutfilePrefix>.chk
        <OutfilePrefix>.csv
        <OutfilePrefix>_Minimization.csv
        <OutfilePrefix>_FinalState.chk
        <OutfilePrefix>_FinalState.xml
        
        <OutfilePrefix>_System.xml
        <OutfilePrefix>_Integrator.xml
        
        <OutfilePrefix>_<DataOutTypePlotY1>Plot.<outExt>
        <OutfilePrefix>_<DataOutTypePlotY2>Plot.<outExt>
        ... ... ...
        
    The reimaged PDB file, <OutfilePrefix>_Reimaged.pdb, corresponds to the first
    frame in the trajectory. The reimaged trajectory file contains all the frames
    aligned to the first frame after reimaging of the frames for periodic systems.

Options:
    -e, --examples
        Print examples.
    -f, --forcefieldParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for biopolymer,
        water, and small molecule forcefields.
        
        The supported parameter names along with their default values are
        are shown below:
            
            biopolymer, amber14-all.xml  [ Possible values: Any Valid value ]
            smallMolecule, openff-2.2.1  [ Possible values: Any Valid value ]
            water, auto  [ Possible values: Any Valid value ]
            additional, none [ Possible values: Space delimited list of any
                valid value ]
            
        Possible biopolymer forcefield values:
            
            amber14-all.xml, amber99sb.xml, amber99sbildn.xml, amber03.xml,
            amber10.xml
            charmm36.xml, charmm_polar_2019.xml
            amoeba2018.xml
        
        Possible small molecule forcefield values:
            
            openff-2.2.1, openff-2.0.0, openff-1.3.1, openff-1.2.1,
            openff-1.1.1, openff-1.1.0,...
            smirnoff99Frosst-1.1.0, smirnoff99Frosst-1.0.9,...
            gaff-2.11, gaff-2.1, gaff-1.81, gaff-1.8, gaff-1.4,...
        
        The default water forcefield valus is dependent on the type of the
        biopolymer forcefield as shown below:
            
            Amber: amber14/tip3pfb.xml
            CHARMM: charmm36/water.xml or None for charmm_polar_2019.xml
            Amoeba: None (Explicit)
            
        Possible water forcefield values:
            
            amber14/tip3p.xml, amber14/tip3pfb.xml, amber14/spce.xml,
            amber14/tip4pew.xml, amber14/tip4pfb.xml,
            charmm36/water.xml, charmm36/tip3p-pme-b.xml,
            charmm36/tip3p-pme-f.xml, charmm36/spce.xml,
            charmm36/tip4pew.xml, charmm36/tip4p2005.xml,
            charmm36/tip5p.xml, charmm36/tip5pew.xml,
            implicit/obc2.xml, implicit/GBn.xml, implicit/GBn2.xml,
            amoeba2018_gk.xml (Implict water)
            None (Explicit water for amoeba)
        
        The additional forcefield value is a space delimited list of any valid
        forcefield values and is passed on to the OpenMMForcefields
        SystemGenerator along with the specified forcefield  values for
        biopolymer, water, and mall molecule. Possible additional forcefield
        values are:
            
            amber14/DNA.OL15.xml amber14/RNA.OL3.xml
            amber14/lipid17.xml amber14/GLYCAM_06j-1.xml
            ... ... ...
            
        You may specify any valid forcefield names supported by OpenMM. No
        explicit validation is performed.
    --freezeAtoms <yes or no>  [default: no]
        Freeze atoms during a simulation. The specified atoms are kept completely
        fixed by setting their masses to zero. Their positions do not change during
        local energy minimization and MD simulation, and they do not contribute
        to the kinetic energy of the system.
    --freezeAtomsParams <Name,Value,..>
        A comma delimited list of parameter name and value pairs for freezing
        atoms during a simulation. You must specify these parameters for 'yes'
        value of '--freezeAtoms' option.
        
        The supported parameter names along with their default values are
        are shown below:
            
            selection, none [ Possible values: CAlphaProtein, Ions, Ligand,
                Protein, Residues, or Water ]
            selectionSpec, auto [ Possible values: A space delimited list of
                residue names ]
            negate, no [ Possible values: yes or no ]
            
        A brief description of parameters is provided below:
            
            selection: Atom selection to freeze.
            selectionSpec: A space delimited list of residue names for
                selecting atoms to freeze. You must specify its value during
                'Ligand' and 'Protein' value for 'selection'. The default values
                are automatically set for 'CAlphaProtein', 'Ions', 'Protein',
                and 'Water' values of 'selection' as shown below:
                
                CAlphaProtein: List of stadard protein residues from pdbfixer
                    for selecting CAlpha atoms.
                Ions: Li Na K Rb Cs Cl Br F I
                Water: HOH
                Protein: List of standard protein residues from pdbfixer.
                
            negate: Negate atom selection match to select atoms for freezing.
            
        In addition, you may specify an explicit space delimited list of residue
        names using 'selectionSpec' for any 'selection". The specified residue
        names are appended to the appropriate default values during the
        selection of atoms for freezing.
    -h, --help
        Print this help message.
    -i, --infile <infile>
        Input file name containing a macromolecule.
    --integratorParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for integrator
        during a simulation.
        
        The supported parameter names along with their default values are
        are shown below:
            
            integrator, LangevinMiddle [ Possible values: LangevinMiddle,
                Langevin, NoseHoover, Brownian ]
            
            randomSeed, auto [ Possible values: > 0 ]
            
            frictionCoefficient, 1.0 [ Units: 1/ps ]
            stepSize, auto [ Units: fs; Default value: 4 fs during yes value of
                hydrogen mass repartioning with no freezing/restraining of atoms;
                otherwsie, 2 fs ] 
            
            barostat, MonteCarlo [ Possible values: MonteCarlo or
                MonteCarloMembrane ]
            barostatInterval, 25
            pressure, 1.0 [ Units: atm ]
            
            Parameters used only for MonteCarloMembraneBarostat with default
            values corresponding to Amber forcefields:
            
            surfaceTension, 0.0 [ Units: atm*A. It is automatically converted 
                into OpenMM default units of atm*nm before its usage.  ]
            xymode,  Isotropic [ Possible values: Anisotropic or  Isotropic ]
            zmode,  Free [ Possible values: Free or  Fixed ]
            
        A brief description of parameters is provided below:
            
            integrator: Type of integrator
            
            randomSeed: Random number seed for barostat and integrator. Not
                supported for NoseHoover integrator.
            
            frictionCoefficient: Friction coefficient for coupling the system to
                the heat bath..
            stepSize: Simulation time step size.
            
            barostat: Barostat type.
            barostatInterval: Barostat interval step size during NPT
                simulation for applying Monte Carlo pressure changes.
            pressure: Pressure during NPT simulation. 
            
            Parameters used only for MonteCarloMembraneBarostat:
            
            surfaceTension: Surface tension acting on the system.
            xymode: Behavior along X and Y axes. You may allow the X and Y axes
                to vary independently of each other or always scale them by the same
                amount to keep the ratio of their lengths constant.
            zmode: Beahvior along Z axis. You may allow the Z axis to vary
                independently of the other axes or keep it fixed.
            
    -m, --mdProtocolParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for executing
        MD protocol.
        
        The supported parameter names along with their default values are
        are shown below:
            
            Phase1 - Initial heating parameters (NVT simulation):
            
            phase1, yes [ Possible values: yes or no ]
            phase1InitialStart, 0.0  [ Units: kelvin ]
            phase1InitialEnd, 300.0  [ Units: kelvin ]
            phase1InitialChange, 5.0  [ Units: kelvin ]
            phase1InitialSteps, 5000
            
            phase1InitialEquilibrationSteps, 100000
            
            Phase2 - Heating and cooling cycles parameters (NVT simulation):
            
            phase2, yes [ Possible values: yes or no ]
            phase2Cycles, 1
            phase2CycleStart, auto  [ Units: kelvin. The default value is set to
                initialEnd ]
            phase2CycleEnd, 315.0  [ Units: kelvin ]
            phase2CycleChange, 1.0  [ Units: kelvin ]
            phase2CycleSteps, 1000
            
            phase2CycleEquilibrationSteps, 100000
            
            Phase3 - NVT equilibration parameters:
            
            phase3, yes [ Possible values: yes or no ]
            phase3Steps, 200000
            
            Phase4 - NPT equilibration parameters:
            
            phase4, yes [ Possible values: yes or no ]
            phase4Steps, 200000
            
            Phase5 - NPT production parameters:
            
            phase5, yes [ Possible values: yes or no ]
            phase5Steps, 1000000
            phase5StepSize, auto [ Units: fs; Default value: Same as stepSize
                parameter in integratorParams option. ]
            
        A brief description of parameters is provided below:
            
            Phase1 - Initial heating parameters (NVT simulation):
            
            phase1: Execute phase1.
            phase1InitialStart: Start temperature for initial heating.
            phase1InitialEnd: End temperature for initial heating.
            phas1InitialChange: Temperature change for increasing temperature
                during initial heating.
            phase1InitialSteps: Number of simulation steps after each
                heating step during initial heating
            
            phase1InitialEquilibrationSteps: Number of equilibration steps
                after the completion of initial heating.
            
            Phase2 - Heating and cooling cycles parameters (NVT simulation):
            
            phase2: Execute phase2.
            phase2Cycles: Number of annealing cycles to perform. Each cycle
                consists of a heating and a cooling phase. The heating phase
                consists of the following steps: Heat system from start to
                end temperature using change size and perform simulation for a
                number of steps after each increase in temperature; Perform
                equilibration after the completion of heating. The cooling
                phase is reverse of the heating phase and cools the system
                from end to start temperature.
            
            phase2CycleStart: Start temperature for annealing cycle.
            phase2CycleEnd: End temperature for annealing cycle.
            phase2CycleChange: Temperature change for increasing or decreasing
                temperature during annealing cycle.
            phase2CycleSteps: Number of simulation steps after each heating and
                cooling step during annealing cycle.
            
            phase2CycleEquilibrationSteps: Number of equilibration steps
                after the completion of heating and cooling phase during a
                annealing cycle.
            
            Phase3 - NVT equilibration parameters:
            
            phase3: Execute phase3.
            phase3Steps: Number of NVT equilibration steps.
            
            Phase4 - NPT equilibration parameters:
            
            phase4: Execute phase4.
            phase4Steps: Number of NPT equilibration steps.
            
            Phase5 - NPT production parameters:
            
            phase5: Execute phase5
            phase5Steps: Number of NPT production steps.
            phase5StepSize: Simulation time step size for NPT production.
            
    -o, --outfileDir <outfiledir>
        Output files directory.
    --outfilePrefix <text>  [default: auto]
        File prefix for generating the names of output files. The default value
        depends on the names of input files for macromolecule and small molecule
        along with the type of statistical ensemble and the nature of the solvation.
        
        The possible values for outfile prefix are shown below:
            
            <InfileRoot>_<Mode>
            <InfileRoot>_Solvated_<Mode>
            <InfileRoot>_<SmallMolFileRoot>_Complex
            <InfileRoot>_<SmallMolFileRoot>_Complex_Solvated
            
    --outputParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        output during a simulation.
        
        The supported parameter names along with their default values are
        are shown below:
            
            checkpoint, no  [ Possible values: yes or no ]
            checkpointFile, auto  [ Default: <OutfilePrefix>.chk ]
            checkpointSteps, 10000
            
            dataOutType, auto [ Possible values: A space delimited list of valid
                parameter names.
                Default: Density Step Speed PotentialEnergy Temperature Time
                    Volume
                Other valid names: ElapsedTime Progress RemainingTime
                KineticEnergy TotalEnergy  ]
            
            dataLog, yes  [ Possible values: yes or no ]
            dataLogFile, auto  [ Default: <OutfilePrefix>.csv ]
            dataLogSteps, 1000
            
            dataStdout, no  [ Possible values: yes or no ]
            dataStdoutSteps, 1000
            
            dataOutTypePlot, yes  [ Possible values: yes or no ]
            dataOutTypePlotX, auto  [ Default: Time; Possible values: Step or
                Time ]
            dataOutTypePlotY, auto  [ Possible values: A space delimited list
                of valid parameter names specified for dataOutType.
                Default: Density PotentialEnergy Temperature Volume
                Other valid names: KineticEnergy TotalEnergy]
            
            minimizationDataSteps, 100
            minimizationDataStdout, no  [ Possible values: yes or no ]
            minimizationDataLog, no  [ Possible values: yes or no ]
            minimizationDataLogFile, auto  [ Default:
                <OutfilePrefix>_MinimizationOut.csv ]
            minimizationDataOutType, auto [ Possible values: A space delimited
                list of valid parameter names.  Default: SystemEnergy
                RestraintEnergy MaxConstraintError.
                Other valid names: RestraintStrength ]
            
            pdbOutFormat, PDB  [ Possible values: PDB or CIF ]
            pdbOutKeepIDs, yes  [ Possible values: yes or no ]
            
            pdbOutMinimized, yes  [ Possible values: yes or no ]
            pdbOutFinal, yes  [ Possible values: yes or no ]
            
            pdbOutPhase1HeatedNVT, yes  [ Possible values: yes or no ]
            pdbOutPhase2AnnealedNVT, yes  [ Possible values: yes or no ]
            pdbOutPhase3EquilibratedNVT, yes  [ Possible values: yes or no ]
            pdbOutPhase4EquilibratedNPT, yes  [ Possible values: yes or no ]
            pdbOutPhase5ProductionNPT, yes  [ Possible values: yes or no ]
            
            saveFinalStateCheckpoint, yes  [ Possible values: yes or no ]
            saveFinalStateCheckpointFile, auto  [ Default:
                <OutfilePrefix>_FinalState.chk ]
            saveFinalStateXML, no  [ Possible values: yes or no ]
            saveFinalStateXMLFile, auto  [ Default:
                <OutfilePrefix>_FinalState.xml]
            
            traj, yes  [ Possible values: yes or no ]
            trajFile, auto  [ Default: <OutfilePrefix>.<TrajFormat> ]
            trajFormat, DCD  [ Possible values: DCD or XTC ]
            trajSteps, 10000 [ The default value corresponds to 40 ps for step
                size of 4 fs. ]
            
            xmlSystemOut, no  [ Possible values: yes or no ]
            xmlSystemFile, auto  [ Default: <OutfilePrefix>_System.xml ]
            xmlIntegratorOut, no  [ Possible values: yes or no ]
            xmlIntegratorFile, auto  [ Default: <OutfilePrefix>_Integrator.xml ]
            
        A brief description of parameters is provided below:
            
            checkpoint: Write intermediate checkpoint file.
            checkpointFile: Intermediate checkpoint file name.
            checkpointSteps: Frequency of writing intermediate checkpoint file.
            
            dataOutType: Type of data to write to stdout and log file.
            
            dataLog: Write data to log file.
            dataLogFile: Data log file name.
            dataLogSteps: Frequency of writing data to log file.
            
            dataStdout: Write data to stdout.
            dataStdoutSteps: Frequency of writing data to stdout.
            
            dataOutTypePlot: Generate plots using data written to log file.
            dataOutTypePlotX: Data out type to plot on X axis.
            dataOutTypePlotY: Data out types to plot on Y axis. An individual plot
                is generated for each pair of X and Y vaues to be plotted.
            
            minimizationDataSteps: Frequency of writing data to stdout
                and log file.
            minimizationDataStdout: Write data to stdout.
            minimizationDataLog: Write data to log file.
            minimizationDataLogFile: Data log fie name.
            minimizationDataOutType: Type of data to write to stdout
                and log file.
            
            saveFinalStateCheckpoint: Save final state checkpoint file.
            saveFinalStateCheckpointFile: Name of final state checkpoint file.
            saveFinalStateXML: Save final state XML file.
            saveFinalStateXMLFile: Name of final state XML file.
            
            pdbOutFormat: Format of output PDB files.
            pdbOutKeepIDs: Keep existing chain and residue IDs.
            
            pdbOutMinimized: Write PDB file after minimization.
            pdbOutFinal: Write final PDB file.
            
            pdbOutPhase1HeatedNVT: Write out PDB file after initial heatin
            pdbOutPhase2AnnealedNVT: Write out PDB file after heating and
                cooling.
            pdbOutPhase3EquilibratedNVT: Write out PDB file after equilibration.
            pdbOutPhase4EquilibratedNPT: Write out PDB file after equilibration.
            pdbOutPhase5ProductionNPT: Write out PDB file after production run.
            
            traj: Write out trajectory file.
            trajFile: Trajectory file name.
            trajFormat: Trajectory file format.
            trajSteps: Frequency of writing trajectory file.
            
            xmlSystemOut: Write system XML file.
            xmlSystemFile: System XML file name.
            xmlIntegratorOut: Write integrator XML file.
            xmlIntegratorFile: Integrator XML file name.
            
    --outPlotParams <Name,Value,...>  [default: auto]
        A comma delimited list of parameter name and value pairs for generating
        plots using Seaborn module. The supported parameter names along with their
        default values are shown below:
            
            type,linepoint,outExt,svg,width,10,height,5.6,
            titleWeight,bold,labelWeight,bold, style,darkgrid,
            palette,deep,font,sans-serif,fontScale,1,
            context,notebook
            
        Possible values:
            
            type: linepoint, scatter, or line. Both points and lines are drawn
                for linepoint plot type.
            outExt: Any valid format supported by Python module Matplotlib.
                For example: PDF (.pdf), PNG (.png), PS (.ps), SVG (.svg)
            titleWeight, labelWeight: Font weight for title and axes labels.
                Any valid value.
            style: darkgrid, whitegrid, dark, white, ticks
            palette: deep, muted, pastel, dark, bright, colorblind
            font: Any valid font name
            context: paper, notebook, talk, poster, or any valid name
            
    --outputReportersMode <text>  [default: ProductionPhaseOnly]
        Add output reporters for production phase only or for all phases of the MD
        protocol. Possible values: AllPhases or ProductionPhaseOnly. The following
        reporters may be added based on the values of correponding output
        parameters specified using '--outputParams' option: TrajReporter,
        DataLogReporter, DataStdoutReporter, and CheckpointReporter.
    --overwrite
        Overwrite existing files.
    -p, --platform <text>  [default: CPU]
        Platform to use for running MD simulation. Possible values: CPU, CUDA,
        OpenCL, or Reference.
    --platformParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs to configure
        platform for running MD simulation.
        
        The supported parameter names along with their default values for
        different platforms are shown below:
            
            CPU:
            
            threads, 1  [ Possible value: >= 0 or auto.  The value of 'auto'
                or zero implies the use of all available CPUs for threading. ]
            
            CUDA:
            
            deviceIndex, auto  [ Possible values: 0, '0 1' etc. ]
            deterministicForces, auto [ Possible values: yes or no ]
            precision, single  [ Possible values: single, double, or mix ]
            tempDirectory, auto [ Possible value: DirName ]
            useBlockingSync, auto [ Possible values: yes or no ]
            useCpuPme, auto [ Possible values: yes or no ]
            
            OpenCL:
            
            deviceIndex, auto  [ Possible values: 0, '0 1' etc. ]
            openCLPlatformIndex, auto  [ Possible value: Number]
            precision, single  [ Possible values: single, double, or mix ]
            useCpuPme, auto [ Possible values: yes or no ]
            
        A brief description of parameters is provided below:
            
            CPU:
            
            threads: Number of threads to use for simulation.
            
            CUDA:
            
            deviceIndex: Space delimited list of device indices to use for
                calculations.
            deterministicForces: Generate reproducible results at the cost of a
                small decrease in performance.
            precision: Number precision to use for calculations.
            tempDirectory: Directory name for storing temporary files.
            useBlockingSync: Control run-time synchronization between CPU and
                GPU.
            useCpuPme: Use CPU-based PME implementation.
            
            OpenCL:
            
            deviceIndex: Space delimited list of device indices to use for
                simulation.
            openCLPlatformIndex: Platform index to use for calculations.
            precision: Number precision to use for calculations.
            useCpuPme: Use CPU-based PME implementation.
            
    --restraintAtoms <yes or no>  [default: no]
        Restraint atoms during a simulation. The motion of specified atoms is
        restricted by adding a harmonic force that binds them to their starting
        positions. The atoms are not completely fixed unlike freezing of atoms.
        Their motion, however, is restricted and they are not able to move far away
        from their starting positions during local energy minimization and MD
        simulation.
    --restraintAtomsParams <Name,Value,..>
        A comma delimited list of parameter name and value pairs for restraining
        atoms during a simulation. You must specify these parameters for 'yes'
        value of '--restraintAtoms' option.
        
        The supported parameter names along with their default values are
        are shown below:
            
            selection, none [ Possible values: CAlphaProtein, Ions, Ligand,
                Protein, Residues, or Water ]
            selectionSpec, auto [ Possible values: A space delimited list of
                residue names ]
            negate, no [ Possible values: yes or no ]
            
        A brief description of parameters is provided below:
            
            selection: Atom selection to restraint.
            selectionSpec: A space delimited list of residue names for
                selecting atoms to restraint. You must specify its value during
                'Ligand' and 'Protein' value for 'selection'. The default values
                are automatically set for 'CAlphaProtein', 'Ions', 'Protein',
                and 'Water' values of 'selection' as shown below:
                
                CAlphaProtein: List of stadard protein residues from pdbfixer
                    for selecting CAlpha atoms.
                Ions: Li Na K Rb Cs Cl Br F I
                Water: HOH
                Protein: List of standard protein residues from pdbfixer.
                
            negate: Negate atom selection match to select atoms for freezing.
            
        In addition, you may specify an explicit space delimited list of residue
        names using 'selectionSpec' for any 'selection". The specified residue
        names are appended to the appropriate default values during the
        selection of atoms for restraining.
    --restraintSpringConstant <number>  [default: 2.5]
        Restraint spring constant for applying external restraint force to restraint
        atoms relative to their initial positions during 'yes' value of '--restraintAtoms'
        option. Default units: kcal/mol/A**2. The default value, 2.5, corresponds to
        1046.0 kjoules/mol/nm**2. The default value is automatically converted into
        units of kjoules/mol/nm**2 before its usage.
    --simulationParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for simulation.
        
        The supported parameter names along with their default values are
        are shown below:
            
            minimization, yes [ Possible values: yes or no ] 
            minimizationMaxSteps, auto  [ Possible values: >= 0. The value of
                zero implies until the minimization is converged. ]
            minimizationTolerance, 0.24  [ Units: kcal/mol/A. The default value
                0.24, corresponds to OpenMM default of value of 10.04
                kjoules/mol/nm. It is automatically converted into OpenMM
                default units before its usage. ]
            
        A brief description of parameters is provided below:
            
            minimization: Perform minimization before equilibration and
                production run.
            minimizationMaxSteps: Maximum number of minimization steps. The
                value of zero implies until the minimization is converged.
            minimizationTolerance: Energy convergence tolerance during
                minimization.
            
    -s, --smallMolFile <SmallMolFile>
        Small molecule input file name. The macromolecue and small molecule are
        merged for simulation and the complex is written out to a PDB file.
    --smallMolID <text>  [default: LIG]
        Three letter small molecule residue ID. The small molecule ID corresponds
        to the residue name of the small molecule and is written out to a PDB file
        containing the complex.
    --systemParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs to configure
        a system for simulation.
        
        The supported parameter names along with their default values are
        are shown below:
            
            constraints, BondsInvolvingHydrogens [ Possible values: None,
                WaterOnly, BondsInvolvingHydrogens, AllBonds, or
                AnglesInvolvingHydrogens ]
            constraintErrorTolerance, 0.000001
            ewaldErrorTolerance, 0.0005
            
            nonbondedMethodPeriodic, PME [ Possible values: NoCutoff,
                CutoffNonPeriodic, or PME ]
            nonbondedMethodNonPeriodic, NoCutoff [ Possible values:
                NoCutoff or CutoffNonPeriodic]
            nonbondedCutoff, 1.0 [ Units: nm ]
            
            hydrogenMassRepartioning, yes [ Possible values: yes or no ]
            hydrogenMass, 1.5 [ Units: amu]
            
            removeCMMotion, yes [ Possible values: yes or no ]
            rigidWater, auto [ Possible values: yes or no. Default: 'No' for
                'None' value of constraints; Otherwise, yes ]
            
        A brief description of parameters is provided below:
            
            constraints: Type of system constraints to use for simulation. These
                constraints are different from freezing and restraining of any
                atoms in the system.
            constraintErrorTolerance: Distance tolerance for constraints as a
                fraction of the constrained distance.
            ewaldErrorTolerance: Ewald error tolerance for a periodic system.
            
            nonbondedMethodPeriodic: Nonbonded method to use during the
                calculation of long range interactions for a periodic system.
            nonbondedMethodNonPeriodic: Nonbonded method to use during the
                calculation of long range interactions for a non-periodic system.
            nonbondedCutoff: Cutoff distance to use for long range interactions
                in both perioidic non-periodic systems.
            
            hydrogenMassRepartioning: Use hydrogen mass repartioning. It
                increases the mass of the hydrogen atoms attached to the heavy
                atoms and decreasing the mass of the bonded heavy atom to
                maintain constant system mass. This allows the use of larger
                integration step size (4 fs) during a simulation.
            hydrogenMass: Hydrogen mass to use during repartioning.
            
            removeCMMotion: Remove all center of mass motion at every time step.
            rigidWater: Keep water rigid during a simulation. This is determined
                automatically based on the value of 'constraints' parameter.
            
    --waterBox <yes or no>  [default: no]
        Add water box.
    --waterBoxParams <Name,Value,..>  [default: auto]
        A comma delimited list of parameter name and value pairs for adding
        a water box.
        
        The supported parameter names along with their default values are
        are shown below:
            
            model, tip3p [ Possible values: tip3p, spce, tip4pew, tip5p or
                swm4ndp ]
            mode, Padding  [ Possible values: Size or Padding ]
            padding, 1.0
            size, None  [ Possible value: xsize ysize zsize ]
            shape, cube  [ Possible values: cube, dodecahedron, or octahedron ]
            ionPositive, Na+ [ Possible values: Li+, Na+, K+, Rb+, or Cs+ ]
            ionNegative, Cl- [ Possible values: Cl-, Br-, F-, or I- ]
            ionicStrength, 0.0
            
        A brief description of parameters is provided below:
            
            model: Water model to use for adding water box. The van der
                Waals radii and atomic charges are determined using the
                specified water forcefield. You must specify an appropriate
                water forcefield. No validation is performed.
            mode: Specify the size of the waterbox explicitly or calculate it
                automatically for a macromolecule along with adding padding
                around ther macromolecule.
            padding: Padding around a macromolecule in nanometers for filling
                box with water. It must be specified during 'Padding' value of
                'mode' parameter.
            size: A space delimited triplet of values corresponding to water
                size in nanometers. It must be specified during 'Size' value of
                'mode' parameter.
            ionPositive: Type of positive ion to add during the addition of a
                water box.
            ionNegative: Type of negative ion to add during the addition of a
                water box.
            ionicStrength: Total concentration of both positive and negative
                ions to add excluding the ions added to neutralize the system
                during the addition of a water box.
            
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To execute MD simulation protocol for a macromolecule in a PDB file, applying
    system constraints for bonds involving hydrogens along with hydrogen mass
    repartioning, using a step size of 4 fs, performing minimization until it's
    converged, performing phase1 initial heating along for 305,000 steps (1.22 ns)
    along with an equilibration for 100,000 steps (400.00 ps) after the completion
    of iniital heating, performing phase 2one heating and cooling cycle along with
    equilibration for 232,000 steps (928 ps), performing phase3 NVT equilibration
    for 200,000 steps (800.00 ps), performing phase 4NPT equilibration for 200,000
    steps (800.00 ps), performing phase NPT production run for 1,000.000 steps
    (4.00 ns), writing trajectory and data log files every 10,000 steps (40 ps) and
    1,000 steps (4 ps) only during the production run, generating a checkpoint file 
    after the completion of the calculation, and generating various PDB files for the
    system during the calculation, type:

        % OpenMMExecuteMDSimulationProtocol.py -i Sample13.pdb
          -o Sample13OutMDProtocol --waterBox yes

    To run the first example for performing OpenMM simulation using multi-
    threading employing all available CPUs on your machine and generate various
    output files, type:

        % OpenMMExecuteMDSimulationProtocol.py -i Sample13.pdb
          -o Sample13OutMDProtocol --waterBox yes
          --platformParams "threads,0"

    To run the first example for performing OpenMM simulation using CUDA platform
    on your machine and generate various output files, type:

        % OpenMMExecuteMDSimulationProtocol.py -i Sample13.pdb
          -o Sample13OutMDProtocol --waterBox yes
          -p CUDA

    To run the second example for a marcomolecule in a complex with a small
    molecule and generate various output files, type:

        % OpenMMExecuteMDSimulationProtocol.py -i Sample13.pdb
          -o Sample13OutMDProtocol --waterBox yes
          -s Sample13Ligand.sdf
          --platformParams "threads,0"

    To run the second example to reporters for writing trajectory, data log, and
    checkpoint files during all phases of the execution of MD protocol, and
    generate various output files, type:

    To run the second example by skipping phase 2 heating and cooling cycle and
    generate various output files, type:

        % OpenMMExecuteMDSimulationProtocol.py -i Sample13.pdb
          -o Sample13OutMDProtocol --waterBox yes
          -s Sample13Ligand.sdf
          --platformParams "threads,0"
          --outputReportersMode AllPhases
 
    To run the second example by freezing CAlpha atoms in a macromolecule without
    using any system constraints to avoid any issues with the freezing of the same atoms,
    using a step size of 2 fs, and generate various output files, type:

        % OpenMMExecuteMDSimulationProtocol.py -i Sample13.pdb
          -o Sample13OutMDProtocol --waterBox yes
          --freezeAtoms yes --freezeAtomsParams "selection,CAlphaProtein"
          --systemParams "constraints, None"
          --platformParams "threads,0" --integratorParams "stepSize,2"

    To run the second example by restrainting CAlpha atoms in a macromolecule and
    and generate various output files, type:

        % OpenMMExecuteMDSimulationProtocol.py -i Sample13.pdb
          -o Sample13OutMDProtocol --waterBox yes --restraintAtoms yes
          --restraintAtomsParams "selection,CAlphaProtein"
          --platformParams "threads,0" --integratorParams "stepSize,2"

    To run the second example by specifying explict values for various parametres
    and generate various output files, type:

        % OpenMMExecuteMDSimulationProtocol.py -i Sample13.pdb
          -o Sample13OutMDProtocol --waterBox yes
          --mdProtocolParams "phase1, yes, phase1InitialStart, 0.0,
          phase1InitialEnd, 300.0, phase1InitialChange, 5.0,
          phase1InitialSteps,  5000, phase1InitialEquilibrationSteps, 100000,
          phase2, yes,phase2Cycles, 1, phase2CycleStart, auto,
          phase2CycleEnd, 315.0, phase2CycleSteps, 1000,
          phase2CycleEquilibrationSteps, 100000, phase3, yes,
          phase3Steps, 200000, phase4, yes, phase4Steps, 200000,
          phase5, yes, phase5Steps, 1000000"
          -f " biopolymer,amber14-all.xml,smallMolecule, openff-2.2.1,
          water,amber14/tip3pfb.xml"
          --integratorParams "integrator,LangevinMiddle,randomSeed,42,
          stepSize,2,pressure, 1.0"
          --outputParams "checkpoint,yes,dataLog,yes,dataStdout,yes,
          minimizationDataStdout,yes,minimizationDataLog,yes,
          pdbOutFormat,CIF,pdbOutKeepIDs,yes,saveFinalStateCheckpoint, yes,
          traj,yes,xmlSystemOut,yes,xmlIntegratorOut,yes"
          -p CPU --platformParams "threads,0"
          --simulationParams "minimization,yes, minimizationMaxSteps,
          5000,equilibration,yes"
          --systemParams "constraints,BondsInvolvingHydrogens,
          nonbondedMethodPeriodic,PME,nonbondedMethodNonPeriodic,NoCutoff,
          hydrogenMassRepartioning, yes"

Author:
    Manish Sud(msud@san.rr.com)

Acknowledgment:
    Paul Charifson

See also:
    OpenMMPrepareMacromolecule.py, OpenMMPerformMDSimulation.py,
    OpenMMPerformSimulatedAnnealing.py, OpenMMPerformMinimization.py

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
