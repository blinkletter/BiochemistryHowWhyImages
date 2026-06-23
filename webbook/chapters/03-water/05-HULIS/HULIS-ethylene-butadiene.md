# The HULIS app and Huckel MO Theory
This notebook provides an overview of how to use the HULIS app from Aix Marselles Université for estimating the relative energies of the $\pi$ molecular orbitals in $sp^2$ systems.

Hückel molecular orbital theory is a quick and dirty method to use an empirical mathematic technique Empirical means that it works, but is not based on any theory that has a real quantum connection to the real system. It is a matrix algebra method and I will not explain it here. We won't need to do this math because there is an app for that.

## The HULIS app

A group at Aix Marseilles Université in France has published a *Java* application that will perform these calculations for us. We can use the app to build a conjugated $\pi$ system and then calculate the energy levels of the molecular orbitals as well as the magnitude of contribution from each atomic $p$ orbital to each of the molecular orbitals (the "shape" of each MO).

You can obtain the app at [http://m.hulis.free.fr/hulis.html](http://m.hulis.free.fr/hulis.html) for a version that works on phones (html) or [https://ctom-ism2.github.io/hulis/index.shtml](https://ctom-ism2.github.io/hulis/index.shtml) for the full *Java* vesrion that will run on your own computer.

## Ethylene MOs

Ethylene (ethene) can be modeled as two connected carbon atoms in HULIS. Use the `Build` button. Every atom that you draw will be $sp^2$ in HULIS so we will model ethylene and use it as the basis for the separated alkene group in 1,4-pentadiene that was discussed in the textbook.

```{figure} HULIS2/Screenshot2026-06-23at9.58.20AM.png
:alt: HULIS app with ethylene
:width: 250px
:align: center
:class: bg-white mb-3

HULIS app with ethylene
```

The energy levels of the MOs is displayed to the right of the molecule. The results of the calculation can be accessed by pressing the `Results` button in the left-hand blue column.

The results are displayed below.

```{figure} HULIS2/Screenshot2026-06-23at9.58.14AMedited.png
:alt: HULIS HMOT results for ethylene
:width: 250px
:align: center
:class: bg-white mb-3

HULIS HMOT results for ethylene.
```

In the results we see information about the MOs. The energy levels are give: in this case, $-1.0\beta$ and $+1.0\beta$. The value of $\beta$ depends on the system, but would be similar for simple molecules. The coefficients for the contribution of each atom to the $\pi$ MO system are given after the energies. Here we see equal contribution from each atom. In the anti-bonding MO we see the change of sign, indicationg a node between the two atoms. The app calculates the total energy of the system. here we see a stabilization energy of $+2.0\beta$ (2 electrons $\times$ $1.0\beta$ for the bonding MO). Finally, the charge in each atom is calculated. Both are carbon and share electrons equally. the electron desnity is evenly distributed and there is no charge on either atom, in this case.

### Visualizing the MOs

By clicking on the energy levels displayed to thge right of the molecule in the HULIS app window, we can call up visualizations of each MO. Below is the bonding and antibonding MOs of ethylene as expressed by the HULIS app.

```{figure} HULIS2/Screenshot2026-06-23at9.58.27AM.png
:alt: Bonding MO for ethylene
:width: 250px
:align: center
:class: bg-white mb-3

The bonding MO for ethylene. The two contributing $p$ atomic orbitals are shown. They are equal in size (proportional to their coefficient of contribution) and there is no node.
```

```{figure} HULIS2/Screenshot2026-06-23at9.58.35AM.png
:alt: Anti-bonding MO for ethylene
:width: 250px
:align: center
:class: bg-white mb-3

The anti-bonding MO for ethylene. The two contributing $p$ atomic orbitals are shown. They are equal in size (proportional to their coefficient of contribution) but opposite in sign. There is a node and the MO does not represent electron sharing between the atoms. This is the opposite of a covalent bond, hence the name "anti-bonding."
```


### The stabilization energy for 1,4-pentadiene

We are using the ethylene molecule as the model for each separated $\pi$ system in 1,4-pentadiene. The $\pi$ stabilization energy in 1,4-pentadiene is estimated to be twice that for a single athylene group. We can say that it is $+4.0\beta$.

## MOs for butadiene

We can use the HULIS app to calculate the MOs for the conjugated $\pi$ system in butadiene. Below is the HULIS app with butadiene entered.

```{figure} HULIS2/Screenshot2026-06-23at9.51.57AM.png
:alt: HULIS app with butadiene
:width: 250px
:align: center
:class: bg-white mb-3

HULIS app with butadiene
```

```{figure} HULIS2/Screenshot2026-06-23at9.52.16AM.png
:alt: HULIS HMOT results app for butadiene
:width: 250px
:align: center
:class: bg-white mb-3

HULIS HMOT results app for butadiene
```


### The stabilization energy for butadiene

The results show a total stabilization energy of $4.47\beta$. This is about 10% more than in the case of two separated ethylene groups. The conjugated $\pi$ system of butadiene is more stable than two unconjugated alkenes. To benefit from this stabilization, the system must be planar so that all the four $p$ atomic orbitals can combine. Conjugation defines the preferred conformers of butadiene.

### Visualizing the MOs

One observation is that the contribution of each atom is not the same in each MO. The lowest-energy MO has greater contribution in the center while the highest-occupird MO (HOMO) has greater contribution at the ends. When butadiene donates electrons to a strong electrophile, the new bond is formed at a terminal carbon. One reason for this observation is that the initial interactions between the HOMO of the butadiene (the nucleophile) and the LUMO of the electrophile is best where the MO has a higher electron density. the electron density in the HOMO is greatest at the ends.

We can use the HULIS app to visualize the relative contributions form each otom in each of the four $\pi$ MOs.

```{figure} HULIS2/Screenshot2026-06-23at9.53.26AM.png
:alt: Bonding MO for ethylene
:width: 250px
:align: center
:class: bg-white mb-3

The lowest-energy bonding MO for butadiene. There is no node.
```

```{figure} HULIS2/Screenshot2026-06-23at9.53.35AM.png
:alt: HOMO MO for ethylene
:width: 250px
:align: center
:class: bg-white mb-3

The HOMO for butadiene. There is one node. Observe the greater electron density at the terminal carbons in this MO.
```

```{figure} HULIS2/Screenshot2026-06-23at9.53.54AM.png
:alt: LUMO MO for ethylene
:width: 250px
:align: center
:class: bg-white mb-3

The LUMO for butadiene. There are two nodes. Observe the greater contribution from the terminal carbons.
```

```{figure} HULIS2/Screenshot2026-06-23at9.54.09AM.png
:alt: Bonding MO for ethylene
:width: 250px
:align: center
:class: bg-white mb-3

The highest-energy anti-bonding MO for butadiene. There are three nodes.
```

## Acrolein

Acrolein is just like butadiene: it is a four-atom $\pi$ system. However, one of the groups is an aldehyde group. The polarity of this group will result in asymmetry in the $\pi$ MO system. We will see that the filled MOs represent graeter electron density toward the oxygen of the aldehyde and the anti-bonding MOs represent a greate ability to accept electrons at the carbonyl carbon and at the $\beta$-carbon (the terminal carbon, in this case). 

The total stabilization energy is $5.81\beta$. 

Here we see different partial charges. The molecule is polar. Again, we see that the carbonyl carbon and the $\beta$ carbon are slightly psoitively charged and likely to be where a nucleophile would add.

```{figure} HULIS2/Screenshot2026-06-23at9.55.16AM.png
:alt: HULIS app with acrolein
:width: 250px
:align: center
:class: bg-white mb-3

HULIS app with acrolein
```

```{figure} HULIS2/Screenshot2026-06-23at9.55.21AM.png
:alt: HULIS HMOT results app for acrolein
:width: 250px
:align: center
:class: bg-white mb-3

HULIS HMOT results app for acrolein
```

```{figure} HULIS2/Screenshot2026-06-23at9.55.53AM.png
:alt: Bonding MO for acrolein
:width: 250px
:align: center
:class: bg-white mb-3

The lowest-energy bonding MO for acrolein. There is no node. Observe the greater electron density at the oxygen atom in this MO.
```

```{figure} HULIS2/Screenshot2026-06-23at9.56.06AM.png
:alt: HOMO MO for acrolein
:width: 250px
:align: center
:class: bg-white mb-3

The HOMO for acrolein. There is one node. 
```

```{figure} HULIS2/Screenshot2026-06-23at9.56.16AM.png
:alt: LUMO MO for acrolein
:width: 250px
:align: center
:class: bg-white mb-3

The LUMO for acrolein. There are two nodes. Observe the greater contribution at the carbonyl carbon and the $\beta$ carbon.
```

```{figure} HULIS2/Screenshot2026-06-23at9.56.25AM.png
:alt: Bonding MO for acrolein
:width: 250px
:align: center
:class: bg-white mb-3

The highest-energy anti-bonding MO for acrolein. There are three nodes.
```

### Relative stabilization in acrolein

We can compare the conjugated acrolein system to the separeate ethylene and formaldehyde systems for estimating the effect of conjugation. We know from above that the stabilization energy for the ethylene $\pi$ system is $2.0\beta$. We can model the formaldehyed $\pi$ system by changing one of the atoms in ethylene from a =CH$_2$ group to a =O group using the `change` button. 

Below are the results. We see that the $\pi$ stabilization energy in formaldehyde is $3.3\beta$. Adding tghe energy for ethylene of $2.0\beta$ gives a total stabilization for the separated groups of $5.3\beta$. Hompare this to the stabilization of $5.81\beta$ calculated above for the conjugated system in acrolein. In the case of acrolein, conjugation appears to increase stabilization energy by $0.51\beta$. This is slightly greater than in the case of butadiene. polar systems benefit more from conjugation. If were were to hand-wave and explanation, we could say that the Lewis structure resonance contrbutors make a little more sense. There is no reason to split electrons between equal carbon atoms (butadiene) but oxygen wants more electrons and so it is more reasonable to draw a resonance contributor that expresses this idea. The real reason is the energies of the molecular orbitals, but Lewis structures are easier to draw and convey the same idea.

```{figure} HULIS2/Screenshot2026-06-23at9.59.36AM.png
:alt: HULIS app with formaldehyde
:width: 250px
:align: center
:class: bg-white mb-3

HULIS app with formaldehyde
```

```{figure} HULIS2/Screenshot2026-06-23at9.59.16AM.png
:alt: HULIS HMOT results app for formaldehyde
:width: 250px
:align: center
:class: bg-white mb-3

HULIS HMOT results app for formaldehyde
```

```{figure} HULIS2/Screenshot2026-06-23at9.59.56AM.png
:alt: Bonding MO for formaldehyde
:width: 250px
:align: center
:class: bg-white mb-3

The lowest-energy bonding MO for formaldehyde. There is no node. Observe the greater electron density at the oxygen atom in this MO.
```

```{figure} HULIS2/Screenshot2026-06-23at10.00.06AM.png
:alt: Anti-bonding MO for formaldehyde
:width: 250px
:align: center
:class: bg-white mb-3

The anti-bonding MO for formaldehyde. There is one node. Observe the greater electron density at the carbon atom in this MO.
```

## Conjugation in amide bonds

Amides are highly polar and have extensive delocalization of electrons between the three heavy atoms of the group. 

### Nitrogen energy

We can model an unconjugated system by adding up the $\pi$ energy for formaldehyde and an $sp^2$ nitrogen. Below are the rsults for the lone nitrogen $p$ atomic orbital. The energy of the $p$ orbital on its own in $2.74\beta$. If were to add this energy to that of an imaginary unconjugated formaldehyde $\pi$ system ($3.30\beta$), we would have a total $\pi$ energy of $6.04\beta$.

```{figure} HULIS2/Screenshot2026-06-23at10.03.17AM.png
:alt: HULIS app with sp2 nitrogen
:width: 250px
:align: center
:class: bg-white mb-3

HULIS app with $sp^2$ nitrogen
```

```{figure} HULIS2/Screenshot2026-06-23at10.03.11AM.png
:alt: HULIS HMOT results app for sp2 nitrogen
:width: 250px
:align: center
:class: bg-white mb-3

HULIS HMOT results app for $sp^2$ nitrogen
```

```{figure} HULIS2/Screenshot2026-06-23at10.03.26AM.png
:alt: p AO for sp2 nitrogen
:width: 250px
:align: center
:class: bg-white mb-3

The $p$-AO for $sp^2$ nitrogen.
```

### Amide energy

We can model an amide by drawing three carbons in the HULIS app and changing them to and $sp^2$ oxygen and nitrogen. The results are shown below. the total $\pi$ stabilization energy is $6.55\beta$. This is $0.51\beta$ greater than in the model for the unconjugated case. Conjugation to a lone pair can be just as important as conjugation between adjacent $\pi$ systems. the nitrogen must be planar and $sp^2$ hybridized to have this conjugation. the amide nitrogen should be tetrahedral ($sp^3$) like most amine groups, but it is observed to be planar when part of an amide group. Conjugation is more powerful than VSEPR.

There is a high degree of charge separation in the amide. The oxygen atoms is calculated to bear most of the electron density and a partial charge of $-0.58$ with the carbon and nitrogen bearing the matching positive charge. Amides make strong hydrogen bonds netween the positively charged N-H group of one amide and the negatively charged oxygen group of another amide. These hydrogen bonds and the planar nature of the groupn are the defining feature of protein secondary structure.



```{figure} HULIS2/Screenshot2026-06-23at10.00.51AM.png
:alt: HULIS app with an amide
:width: 250px
:align: center
:class: bg-white mb-3

HULIS app with an amide.
```

```{figure} HULIS2/Screenshot2026-06-23at10.00.46AM.png
:alt: HULIS HMOT results app for an amide
:width: 250px
:align: center
:class: bg-white mb-3

HULIS HMOT results app for an amide.
```

```{figure} HULIS2/Screenshot2026-06-23at10.01.02AM.png
:alt: Lowest-energy bonding MO for an amide
:width: 250px
:align: center
:class: bg-white mb-3

Lowest-energy bonding MO for an amide.
```


```{figure} HULIS2/Screenshot2026-06-23at10.01.11AM.png
:alt: HOMO MO for an amide
:width: 250px
:align: center
:class: bg-white mb-3

HOMO for an amide. The node is not exactly on the middle atom (the carbon atom). The group is asymmetrical and so this is not unexpected.
```

```{figure} HULIS2/Screenshot2026-06-23at10.01.21AM.png
:alt: LUMO for an amide
:width: 250px
:align: center
:class: bg-white mb-3

LUMO for an amide. Observe the greatest contribution at the carbon. This is where nucleophiles would attack.
```



% ':class: bg-white mb-3' sets the background of the image to 'white' and the bottom space to 3 units. see  https://getbootstrap.com/docs/4.0/utilities/colors/