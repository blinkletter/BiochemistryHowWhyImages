# Huckel MO Theory
This notebook provides an overview of how to use the HULIS app from Aix Marselles Université for estimating the relative energies of the $\pi$ molecular orbitals in $sp^2$ systems.

Hückel molecular orbital theory is a quick and dirty method to use an empirical mathematic technique Empirical means that it works, but is not based on any theory that has a real quantum connection to the real system. It is a matrix algebra method and I will not explain it here. We won't need to do this math because there is an app for that.

## The HULIS app

A group at Aix Marseilles Université in France has published a *Java* application that will perform these calculations for us. We can use the app to build a conjugated $\pi$ system and then calculate the energy levels of the molecular orbitals as well as the magnitude of contribution from each atomic $p$ orbital to each of the molecular orbitals (the "shape" of each MO).

You can obtain the app at http://m.hulis.free.fr/hulis.html for a version that works on phones (html) or https://ctom-ism2.github.io/hulis/index.shtml for the full *Java* vesrion that will run on your own computer.

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

% ':class: bg-white mb-3' sets the background of the image to 'white' and the bottom space to 3 units. see  https://getbootstrap.com/docs/4.0/utilities/colors/