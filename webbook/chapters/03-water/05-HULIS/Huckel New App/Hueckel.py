#program exemplifying the Hueckel Method
import os
import sys
import numpy as np
from numpy import linalg as LA

#options to control
plotMOs = True
plotMOdiagram = True

#constants
alphaC = -158.0   #kcal/mol
betaCC = -84.0    #kcal/mol
alphaN = alphaC + 1.47*betaCC
alphaO = alphaC + 1.70*betaCC
betaCN = betaCC*0.89
betaCO = betaCC*0.80

def ReadSDF(sdffile):
    #function reading an sdf file to extract
    # 1) the number of atoms, Natoms
    # 2) the list of atoms, AtomList
    # 3) the bonding pattern, Bonding
    rfile = open(sdffile,"r")
    rfilelines = rfile.readlines()
    Natoms = int(rfilelines[3][0:3])
    AtomList = []
    Bonding = np.zeros([Natoms,Natoms])
    geometry = []
    for idatm in range(4,Natoms + 4):
        line = rfilelines[idatm].strip().rstrip().replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ")
        arraydata = line.split(" ")
        AtomList.append(arraydata[3])
        geometry.append([float(arraydata[0]),float(arraydata[1]),float(arraydata[2])])
    np.resize(Bonding,(Natoms,Natoms))
    for idbond in range(Natoms + 4,len(rfilelines)):
        line = rfilelines[idbond].strip().rstrip().replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ")
        if ("M END" in line): break
        arraydata = line.split(" ")
        atom1 = int(arraydata[0]) - 1
        atom2 = int(arraydata[1]) - 1
        bo = int(arraydata[2])
        Bonding[atom1][atom2] = bo
        Bonding[atom2][atom1] = bo
    return Natoms, AtomList, Bonding, geometry

def CleanStructure(Natoms, AtomList, Bonding, geometry):
    #function that removes atoms not in the pi system
    import copy
    new_Natoms = Natoms
    new_AtomList = []
    new_geometry = []
    new_Bonding = copy.deepcopy(Bonding)
    indices2delete = []
    for idatm in range(Natoms):
        maxbo = 1
        #check the max bond order for a given atom
        for idbtm in range(Natoms):
            if Bonding[idatm][idbtm] > maxbo: maxbo = Bonding[idatm][idbtm]
            if maxbo > 1: break
        if (maxbo == 1) and AtomList[idatm] == "C":
            #atom is not in double bond
            indices2delete.append(idatm)
            new_Natoms -= 1
            continue
        if AtomList[idatm] == "H":
            #protons
            indices2delete.append(idatm)
            new_Natoms -= 1
            continue
        new_AtomList.append(AtomList[idatm])
        new_geometry.append(geometry[idatm])
    #remove atoms from BOs
    for idx in range(len(indices2delete)):
        inewdx = indices2delete[len(indices2delete) - 1 - idx]
        new_Bonding2 = np.delete(new_Bonding,inewdx,axis = 0)
        new_Bonding = np.delete(new_Bonding2,inewdx,axis = 1)
    return new_Natoms, new_AtomList, new_Bonding, new_geometry, indices2delete

def ConvertElements2AtomicNumb(ElementList):
    #function converts elements to atomic number
    AtomList = []
    for iel in ElementList:
        if iel == "H": AtomList.append(1)
        elif iel == "C": AtomList.append(6)
        elif iel == "N": AtomList.append(7)
        elif iel == "O": AtomList.append(8)
        else: AtomList.append(iel)
    return AtomList

def CreateHamiltonian(BondPattern, Natoms, AtomList):
    #function creating the Hueckel Hamiltonian from the bonding pattern
    Hamiltonian = np.zeros([Natoms,Natoms])
    for idi in range(Natoms):
        Hamiltonian[idi][idi] = alphaC*(AtomList[idi] == 6) + alphaN*(AtomList[idi] == 7) + alphaO*(AtomList[idi] == 8)
        for idj in range(idi + 1,Natoms):
            carbon = (AtomList[idi] == 6) or (AtomList[idj] == 6)
            nitrogen = (AtomList[idi] == 7) or (AtomList[idj] == 7)
            oxygen = (AtomList[idi] == 8) or (AtomList[idj] == 8)
            value = ((BondPattern[idi][idj] == 2) + (BondPattern[idi][idj] == 4))*(betaCC*(carbon)*(not nitrogen)*(not oxygen) + betaCN*(carbon)*(nitrogen)*(not oxygen) + betaCO*(carbon)*(not nitrogen)*(oxygen))
            Hamiltonian[idi][idj] = value
            Hamiltonian[idj][idi] = value
    return Hamiltonian

def DiagonaliseHamiltonian(Hamiltonian):
    MOenergies,MOvectors = LA.eigh(np.array(Hamiltonian))
    return MOenergies, MOvectors

def ResortOrbitals(EMOs, CMOs):
    #resorts the orbitals according to their energy
    indices = sorted(range(len(EMOs)),key = EMOs.__getitem__)
    transCMOs = CMOs.transpose()
    sorted_EMOs = np.array(list(map(list(EMOs).__getitem__,indices)))
    trans_sorted_CMOs = np.array(list(map(list(transCMOs).__getitem__,indices)))
    sorted_CMOs = trans_sorted_CMOs.transpose()
    return sorted_EMOs, sorted_CMOs

def CreateMODiagram(EMOs, CMOs, Atoms, NElectrons, electronic_state, threshold = 2.0):
    #function that writes the MO diagram for the system
    import matplotlib.pyplot as plt
    #get the delta of energies
    deltaE = EMOs[len(EMOs) - 1] - EMOs[0]
    electronsize = deltaE*0.05
    #max degeneracy observed
    maxdegen = 1
    #get the axis
    x_mo = []
    for idmo in range(len(EMOs)):
        x_mo.append(1.0)
    #deal with degeneracy
    setEMOs = list(set(EMOs))
    Degeneracy = np.zeros([len(setEMOs),2])
    for idemo in range(len(setEMOs)):
        Degeneracy[idemo][0] = setEMOs[idemo]
        for idmo in EMOs:
            if np.abs(idmo - Degeneracy[idemo][0]) < threshold: Degeneracy[idemo][1] += 1
        offset = []
        if Degeneracy[idemo][1] == 1.0: continue
        elif Degeneracy[idemo][1] == 2.0: 
            offset = [0.0,0.5]
            if maxdegen < 2: maxdegen = 2
        elif Degeneracy[idemo][1] == 3.0: 
            offset = [0.0,0.5,1.0]
            maxdegen = 3
        else: 
            offset = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0]
            maxdegen = 21
        counter = 0
        for idmo in range(len(EMOs)):
            if np.abs(EMOs[idmo] - Degeneracy[idemo][0]) < threshold:
                x_mo[idmo] += offset[counter]
                counter += 1
    fig, ax = plt.subplots()
    plt.title("MO diagram")
    ax.scatter(x_mo,EMOs,s = 1444,c = "royalblue",marker = "_",linewidth = 3,zorder = 3)
    ax.grid(axis = 'y')
    x_ao = []
    EAOs = []
    orbnames = []
    setatoms = list(set(Atoms))
    for idmo in range(len(setatoms)):
        x_ao.append(0.5)
        if setatoms[idmo] == "C": 
            EAOs.append(alphaC)
            orbnames.append("${p}_{C}$")
        elif setatoms[idmo] == "N": 
            EAOs.append(alphaN)
            orbnames.append("${p}_{N}$")
        elif setatoms[idmo] == "O": 
            EAOs.append(alphaO)
            orbnames.append("${p}_{O}$")
    ax.scatter(x_ao,EAOs,c = "seagreen",s = 1444,marker = "_",linewidth = 3,zorder = 3)
    ax.axes.get_xaxis().set_ticks([])
    ax.margins(0.2)
    #get connection between MOs and AOs
    connection_lines = []
    for idmo in range(len(EMOs)):
        for jdmo in range(len(EMOs)):
            if np.fabs(CMOs[jdmo][idmo]) > 0.001:
                eAO = 0.0
                if Atoms[jdmo] == "C": eAO = alphaC
                elif Atoms[jdmo] == "N": eAO = alphaN
                elif Atoms[jdmo] == "O": eAO = alphaO
                ao_idx = -1
                for idao in range(len(EAOs)):
                    if EAOs[idao] == eAO:
                        ao_idx = idao
                        break
                x_atoms = [x_ao[ao_idx],x_mo[idmo]]
                y_atoms = [EAOs[ao_idx],EMOs[idmo]]
                connection_lines.append([x_atoms,y_atoms])
    for idc in range(len(connection_lines)):
        plt.plot(connection_lines[idc][0],connection_lines[idc][1],'k',linewidth = 1,linestyle = "--")
    #annotate AOS
    for xi,yi,tx in zip(x_ao,EAOs,orbnames):
        ax.annotate(tx,xy = (xi,yi),xytext = (0,-4),size = 18,ha = "center",va = "top",textcoords = "offset points")
    #add label to y axis
    plt.ylabel("Energy (kcal/mol)",size = 24)
    #add electrons to the plot
    maxdx = 0.1/6*maxdegen
    for ismo in range(NElectrons):
        smoidx = ismo
        upwards = True
        if smoidx%2 != 0: 
            smoidx -= 1
            upwards = False
        moidx = int(smoidx/2)
        if (ismo == NElectrons - 1) and (electronic_state != 0):
            moidx += electronic_state
        shiftx = -maxdx
        shifty = -electronsize
        if not upwards: 
            shiftx = maxdx
            shifty = electronsize
        ax.annotate("", xy=(x_mo[moidx] + shiftx,EMOs[moidx] - shifty), xytext=(x_mo[moidx] + shiftx,EMOs[moidx] + shifty),arrowprops = dict(arrowstyle= '-|>',color = 'black',lw = 2.0,ls = '-'))
    fig.tight_layout()
    return fig

def AddCircle(ax, coord, radius, atom_colour):
    #function to add circle to a plot
    import matplotlib.pyplot as plt
    circleA = plt.Circle(coord,radius,color = atom_colour)
    circleB = plt.Circle(coord,radius,color = "black",linewidth = 1,fill = False)
    ax.add_patch(circleA)
    ax.add_patch(circleB)

def PlotPopulations(Natoms, AtomList, geom, BondPattern, populations, atoms2skip):
    #function plotting pi-electron populations
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    plt.title("$\pi$ populations")
    #add bonds
    Bonds = []
    for idatm in range(Natoms):
        for idbtm in range(idatm + 1,Natoms):
            if BondPattern[idatm][idbtm]:
                x_atoms = [geom[idatm][0],geom[idbtm][0]]
                y_atoms = [geom[idatm][1],geom[idbtm][1]]
                Bonds.append([x_atoms,y_atoms])
    for idbond in range(len(Bonds)):
        plt.plot(Bonds[idbond][0],Bonds[idbond][1],color = "black",linestyle = "-")
    #add atoms
    for iatm in range(Natoms):
        coord = [geom[iatm][0],geom[iatm][1]]
        atom_colour = "papayawhip"
        if AtomList[iatm] == "N": atom_colour = "royalblue"
        if AtomList[iatm] == "O": atom_colour = "salmon"
        AddCircle(ax,coord,0.2,atom_colour)
        #annotate charge?
        if iatm not in atoms2skip:
            coordpert = [geom[iatm][0] + 0.1,geom[iatm][1] - 0.25]
            ax.annotate(str(round(populations[iatm],3)),xy = coordpert,xytext = coordpert)
    ax.set_aspect('equal')
    xmin = 100000000.0
    xmax = -100000000.0
    ymin = 100000000.0
    ymax = -100000000.0
    for idatm in range(len(geom)):
        if geom[idatm][0] < xmin: xmin = geom[idatm][0]
        elif geom[idatm][0] > xmax: xmax = geom[idatm][0]
        if geom[idatm][1] < ymin: ymin = geom[idatm][1]
        elif geom[idatm][1] > ymax: ymax = geom[idatm][1]
    ax.set(xlim = (xmin - 1.0,xmax + 1.0),ylim = (ymin - 1.0,ymax + 1.0))
    fig.tight_layout()
    return fig

def CheckPlanarity(Geom, Natoms):
    #check for the flat dimension, since we are only plotting pz orbitals
    Xnot0 = 0
    Ynot0 = 0
    Znot0 = 0
    for idatm in range(Natoms): 
        if np.abs(Geom[idatm][0]) > 0.000001: Xnot0 += 1
        if np.abs(Geom[idatm][1]) > 0.000001: Ynot0 += 1
        if np.abs(Geom[idatm][2]) > 0.000001: Znot0 += 1
    #if there is no dimension with zeroes, go away
    #if (Xnot0 != 0) and (Ynot0 != 0) and (Znot0 != 0): 
    if (Znot0 != 0): 
        print("make the molecule flat on one axis")
        return 1
    return 0

def FixGeometry(Geom, Natoms):
    #check for the flat dimension, since we are only plotting pz orbitals
    Xnot0 = 0
    Ynot0 = 0
    Znot0 = 0
    for idatm in range(Natoms): 
        if np.abs(Geom[idatm][0]) > 0.000001: Xnot0 += 1
        if np.abs(Geom[idatm][1]) > 0.000001: Ynot0 += 1
        if np.abs(Geom[idatm][2]) > 0.000001: Znot0 += 1
    #if there is no dimension with zeroes, go away
    if (Xnot0 != 0) and (Ynot0 != 0) and (Znot0 != 0): 
        print("make the molecule flat on one axis")
        return 1
    #if the zero-dimension is not z, then swap
    NewGeom_blah = []
    if Ynot0 == 0:
        NewGeom = []
        for idatm in range(Natoms):
            aux = [Geom[idatm][0],Geom[idatm][2],Geom[idatm][1]]
            NewGeom.append(aux)
        Geom = NewGeom
        for idatm in range(Natoms):
            aux = [Geom[idatm][0],Geom[idatm][2],Geom[idatm][1]]
            NewGeom_blah.append(aux)
    elif Xnot0 == 0:
        NewGeom = []
        for idatm in range(Natoms):
            aux = [Geom[idatm][1],Geom[idatm][2],Geom[idatm][0]]
            NewGeom.append(aux)
        Geom = NewGeom
        for idatm in range(Natoms):
            aux = [Geom[idatm][1],Geom[idatm][2],Geom[idatm][0]]
            NewGeom_blah.append(aux)
    new_geometry = NewGeom_blah
    return new_geometry

def RunHueckel(argv):
    #getting the system
    molecule = argv[1]
    charge = 0
    if (len(argv) > 2): charge = int(argv[2])
    electronic_state = 0
    if (len(argv) > 3): electronic_state = int(argv[3])
    
    #prepare
    Natoms,ElementList,BondPattern,Geom = ReadSDF(molecule)
    new_Natoms,new_ElementList,new_BondPattern,new_geometry,atoms2skip = CleanStructure(Natoms,ElementList,BondPattern,Geom)
    AtomList = ConvertElements2AtomicNumb(ElementList)
    new_AtomList = ConvertElements2AtomicNumb(new_ElementList)

    #"SCF"
    Hamiltonian = CreateHamiltonian(new_BondPattern,new_Natoms,new_AtomList)
    MOenergies,MOvectors = DiagonaliseHamiltonian(Hamiltonian)
    MOenergies,MOvectors = ResortOrbitals(MOenergies,MOvectors)
    
    #count electrons
    Nelectrons = -charge
    for idatm in range(len(new_AtomList)): 
        n1 = 0
        n2 = 0
        n4 = 0
        neighbours = 0
        for idbtm in range(len(new_AtomList)):
            if new_BondPattern[idatm][idbtm] == 1: n1 += 1
            elif new_BondPattern[idatm][idbtm] == 2: n2 += 1
            elif new_BondPattern[idatm][idbtm] == 4: n4 += 1
            neighbours += (new_BondPattern[idatm][idbtm] != 0)
        if ((n2 == 0) and (n4 == 0)) and ((new_AtomList[idatm] == 7) or (new_AtomList[idatm] == 8)): Nelectrons += 2
        elif (n4 == 3) and (new_AtomList[idatm] == 7): Nelectrons += 2
        elif (n4 == 2) and (n1 == 1) and (new_AtomList[idatm] == 7): Nelectrons += 2
        elif ((n2 == 1) or (n4 >= 1)) and ((new_AtomList[idatm] == 6) or (new_AtomList[idatm] == 7) or (new_AtomList[idatm] == 8)): Nelectrons += 1 + (new_AtomList[idatm] == 8)*((n4 > 1) + (n4 == 1)*(len(new_AtomList)%2 != 0)) + (new_AtomList[idatm] == 8)*(n1 == 2) + (new_AtomList[idatm] == 7)*(n4 == 1)
        if (new_AtomList[idatm] == 8) and (neighbours == 1):
            #determine bond pattern of neighbour
            for idbtm in range(len(new_AtomList)):
                if new_BondPattern[idatm][idbtm] != 0:
                    neighbours = 0
                    nN = 0
                    for idctm in range(len(new_AtomList)):
                        neighbours += (new_BondPattern[idbtm][idctm] == 4)
                        nN += (new_AtomList[idctm] == 7)*(new_BondPattern[idbtm][idctm] == 4)
                    #add extra electron?
                    Nelectrons += (neighbours == 3)
                    Nelectrons -= (nN == 1)*(new_BondPattern[idatm][idbtm] == 4)
                    break
        elif (new_AtomList[idatm] == 8) and (neighbours == 2) and (n1 == 1): Nelectrons += 1
    
    #get spin orbitals
    spinMOs = []
    spinCMOs = []
    for edmo in MOenergies:
        spinMOs.append(edmo)
        spinMOs.append(edmo)
    for cdmo in MOvectors:
        spinCMOs.append(cdmo)
        spinCMOs.append(cdmo)
    
    #get energy and pi electron populations
    Energy = 0.0
    populations = np.zeros(new_Natoms)
    HOMO = Nelectrons - 1
    for ielec in range(Nelectrons):
        Energy += spinMOs[ielec]
        moidx = int(ielec/2)
        for iatm in range(len(populations)):
            populations[iatm] += MOvectors[iatm][moidx]*MOvectors[iatm][moidx]
    
    if electronic_state > 0: 
        if int((HOMO + 1)/2) + electronic_state > len(MOenergies): electronic_state = len(MOenergies) - int((HOMO + 1)/2)
        moidx = int(HOMO/2)
        for iatm in range(len(populations)):
            populations[iatm] -= MOvectors[iatm][moidx]*MOvectors[iatm][moidx]
            populations[iatm] += MOvectors[iatm][moidx + electronic_state]*MOvectors[iatm][moidx + electronic_state]
        Energy -= spinMOs[HOMO]
        Energy += spinMOs[HOMO + electronic_state*2]
    
    return Energy, Natoms, Nelectrons, electronic_state, ElementList, MOenergies, populations, MOvectors, Geom, BondPattern, atoms2skip, new_ElementList

def add_spheres_feature_view(view, feature, xyz, viewnum, sizefactor):
    normalization = max(abs(max(feature)),abs(min(feature)))
    spec_color_plus = "#0509ab"
    spec_color_minus = "#9c1d0b"
    for i in range(len(feature)):
        print(i)
        colorA = spec_color_plus
        colorB = spec_color_minus
        if feature[i] < 0: 
            colorA = spec_color_minus
            colorB = spec_color_plus
        shiftz = 0.75
        size_sphere = 0.75*abs(feature[i])/normalization*sizefactor
        if size_sphere < 0.01: continue
        if size_sphere < 0.5: shiftz = 0.50
        if size_sphere < 0.25: shiftz = 0.25
        view.addSphere({'center':{
        'x':xyz[i][0], 
        'y':xyz[i][1],
        'z':xyz[i][2] + shiftz},
        'radius':size_sphere,'color':colorA,'alpha':1.00},viewer = viewnum) 
        view.addSphere({'center':{
        'x':xyz[i][0], 
        'y':xyz[i][1],
        'z':xyz[i][2] - shiftz},
        'radius':size_sphere,'color':colorB,'alpha':1.00},viewer = viewnum) 
    return view

def plotMOs(Natoms_short, Geom, MOvectors, MOplot, atoms2skip):
    import py3Dmol
    extension = "sdf"
    planarityOK = CheckPlanarity(Geom,Natoms_short)
    #Geom = FixGeometry(Geom, Natoms)
    if planarityOK == 0:
        input_file = open('dummy_struct.' + extension,"r").read()
        view = py3Dmol.view(width = 620,height = 620,viewergrid = (1,2))
        view.setBackgroundColor('white')
        view.addModel(input_file,extension,viewer = (0,0))
        view.addModel(input_file,extension,viewer = (0,1))
        view.setStyle({'stick': {'colorscheme': {'prop': 'resi', 'C': '#2e2b2b'}, "radius":"0.07"}}, viewer=(0,0))
        view.setStyle({'stick': {'colorscheme': {'prop': 'resi', 'C': '#cccccc'}, "radius":"0.07"}}, viewer=(0,1))
        MO2plot = []
        for idatm in range(Natoms_short):
            MO2plot.append(MOvectors[idatm][MOplot - 1])
        if (atoms2skip[0] != -1):
            for idatm in atoms2skip:
                MO2plot.insert(idatm,0.0)
        add_spheres_feature_view(view,MO2plot,Geom,(0,0),1.0)
        view.zoomTo(viewer=(0,0))
        view.zoomTo(viewer=(0,1))
        output = view._make_html().replace("'", '"')
        x = f"""<!DOCTYPE html><html> {output} </html>"""  # do not use ' in this input
        visualization_html = f"""<iframe  style="width: 100%; height:620px" name="result" allow="midi; geolocation; microphone; camera; 
        display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
        allow-scripts allow-same-origin allow-popups 
        allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
        allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>"""   
    return visualization_html
