import gradio as gr
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from Hueckel import *

def get_atom_features(Elements,Populations):
    qm_atom_features = {}
    qm_atom_features['atom types'] = Elements
    for idatm in range(len(Populations)):
        value = round(Populations[idatm],3)
        Populations[idatm] = value
    qm_atom_features['electron population'] = Populations
    return qm_atom_features

def get_mol_features(MOenergies):
    qm_mol_features = {}
    for idorb in range(len(MOenergies)):
        value = round(MOenergies[idorb],3)
        MOenergies[idorb] = value
    qm_mol_features['MO energies'] = MOenergies
    return qm_mol_features

def Save2File(Natoms,geometry,MOvectors,Elements,BondPattern,atoms2skip):
    wfile = open("results.txt","w")
    Natoms_pi = len(MOvectors)
    wfile.write(str(Natoms) + ";" + str(Natoms_pi) + "\n")
    for idatm in range(Natoms):
        wfile.write(str(geometry[idatm][0]) + ";" + str(geometry[idatm][1]) + ";" + str(geometry[idatm][2]) + "\n")
    wfile.write("------------\n")
    for idatm in range(Natoms_pi):
        strng = ""
        for idbtm in range(Natoms_pi - 1):
            strng += str(MOvectors[idatm][idbtm]) + ";"
        strng += str(MOvectors[idatm][Natoms_pi - 1])
        wfile.write(strng + "\n")
    wfile.write("------------\n")
    strng = ""
    for idatm in range(Natoms - 1):
        strng += Elements[idatm] + ";"
    strng += Elements[Natoms - 1]
    wfile.write(strng + "\n")
    wfile.write("------------\n")
    for idatm in range(Natoms):
        strng = ""
        for idbtm in range(Natoms - 1):
            strng += str(BondPattern[idatm][idbtm]) + ";"
        strng += str(BondPattern[idatm][Natoms - 1])
        wfile.write(strng + "\n")
    wfile.write("------------\n")
    strng = ""
    for idatm in range(len(atoms2skip) - 1):
        strng += str(atoms2skip[idatm]) + ";"
    if (len(atoms2skip) > 0): strng += str(atoms2skip[len(atoms2skip) - 1])
    else: strng += "-1"
    wfile.write(strng + "\n")
    wfile.write("------------\n")

def ReadSaveFile():
    Natoms = 0
    geometry = []
    MOvectors = []
    Elements = []
    BondPattern = []
    atoms2skip = []
    rfile = open("results.txt","r")
    rfilelines = rfile.readlines()
    firstdata = rfilelines[0].split(";")
    Natoms = int(firstdata[0])
    Natoms_pi = int(firstdata[1])
    #get geometry
    for idatm in range(Natoms):
        line = rfilelines[idatm + 1].split(";")
        aux = [float(line[0]),float(line[1]),float(line[2])]
        geometry.append(aux)
    lastline = Natoms + 2
    #get the MO vector
    for idatm in range(Natoms_pi):
        line = rfilelines[lastline + idatm].split(";")
        aux = []
        for idbtm in range(len(line)):
            aux.append(float(line[idbtm]))
        MOvectors.append(aux)
    #get the element list
    lastline = Natoms + Natoms_pi + 3
    Elements = rfilelines[lastline].split(";")
    #get the bond pattern
    lastline = Natoms + Natoms_pi + 5
    for idatm in range(Natoms):
        line = rfilelines[lastline + idatm].split(";")
        aux = []
        for idbtm in range(len(line)):
            aux.append(float(line[idbtm]))
        BondPattern.append(aux)
    #get the list of atoms to skip
    lastline = 2*Natoms + Natoms_pi + 6
    aux = rfilelines[lastline].split(";")
    atoms2skip = []
    for idx in range(len(aux)): 
        atoms2skip.append(int(aux[idx]))
    return Natoms,np.array(geometry),np.array(MOvectors),Elements,np.array(BondPattern),atoms2skip

def runHueckel(input_file, charge, estate, threshold = 2.0):
    input_f = open(input_file.name, "r").read()
    input_format = input_file.name.split('.')[-1]
    with open('dummy_struct.' + input_format, "w") as oF:
        oF.write(input_f)
    #prepare
    Hueckel_input = ["filename","dummy_struct." + input_format,charge,estate]
    #calculate
    Energy,Natoms,Nelectrons,electronic_state,Elements,MOenergies,Populations,MOvectors,geometry,BondPattern,atoms2skip,new_ElementList = RunHueckel(Hueckel_input)
    #store/process
    pops = list(Populations)
    for idatm in atoms2skip:
        if idatm > len(Populations): pops.append(0.0)
        else: pops.insert(idatm,0.0)
    Populations = np.array(pops)
    atom_features = get_atom_features(Elements,Populations)
    molecular_features = get_mol_features(MOenergies)
    MOdiagram = CreateMODiagram(MOenergies,MOvectors,new_ElementList,Nelectrons,electronic_state,threshold)
    ChargeDiagram = PlotPopulations(Natoms,Elements,geometry,BondPattern,Populations,atoms2skip)
    Save2File(Natoms,geometry,MOvectors,Elements,BondPattern,atoms2skip)
    return round(Energy,3),Natoms,Nelectrons,electronic_state,pd.DataFrame(atom_features),pd.DataFrame(molecular_features),MOdiagram,ChargeDiagram

def get_molecular_features(Energy,Natoms,charge,Nelectrons):
    mol_features = {}
    mol_features['Energy'] = [Energy]
    mol_features['Natoms'] = [Natoms]
    mol_features['Charge'] = [charge]
    mol_features['Nelectrons'] = [Nelectrons]
    return pd.DataFrame(mol_features)

def MOPlotter(mo2plot):
    #function to plot MOs
    Natoms,geometry,MOvectors,Elements,BondPattern,atoms2skip = ReadSaveFile()
    Natoms_pi = len(MOvectors)
    MOplot = plotMOs(Natoms_pi,geometry,MOvectors,mo2plot,atoms2skip)
    return MOplot

with gr.Blocks() as demo:
    gr.Markdown("# Hueckel Molecular Orbital")
    with gr.Row():
        input_file = gr.File(label = "input flat-molecule SDF file")
        input_file2 = input_file
        charge = gr.Textbox(placeholder = "Total charge",label = "System's charge.")
        estate = gr.Textbox(placeholder = "Electronic State",label = "System's electronic state.")
    
    main_btn = gr.Button(value = "Run")

    with gr.Row():
        basic_html = gr.HTML()
    
    with gr.Row():
        with gr.Column(scale = 1):
            energy = gr.Number(label="Energy (kcal/mol)")
            natoms = gr.Number(label="Number of atoms")
            nelectrons = gr.Number(label="Number of electrons")
            electronic_state = gr.Number(label="Electronic State")
        MOdiagram = gr.Plot(scale = 2)
    with gr.Row():
        df_atom_features = gr.Dataframe(label = "Atomic Properties")
        df_mol_features = gr.Dataframe(label = "Molecular Properties")
        CHARGEdiagram = gr.Plot()
    
    main_btn.click(fn = runHueckel,inputs = [input_file,charge,estate], outputs = [energy,natoms,nelectrons,electronic_state,df_atom_features,df_mol_features,MOdiagram,CHARGEdiagram])

    with gr.Row():
        with gr.Column():
            mo2plot = gr.Number(label = "Plot an MO")
            MO_btn = gr.Button(value = "Plot MO")
        MOrepresentation = gr.HTML()
    
    MO_btn.click(fn = MOPlotter,inputs = [mo2plot], outputs = [MOrepresentation])

    demo.launch(server_name = "0.0.0.0",server_port = 7860)
